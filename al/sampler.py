from abc import ABC, abstractmethod
import numpy as np
import torch

from dataloaders import make_iterable


class Sampler(ABC):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def query(self, query_size, *args, **kwargs):
        pass

    def _forward_iter(self, indices, forward_fn):
        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )
        out_list = []
        for batch in iter:
            ret_val = batch.text
            if type(ret_val) is tuple:
                # Unpack inputs and lengths.
                x, lengths = ret_val
            else:
                x = ret_val
                # Set lengths to None to match the method's signature.
                lengths = None
            out = forward_fn(x, lengths=lengths)
            out_list.append(out)

        res = torch.cat(out_list)
        return res

    def _predict_probs_dropout(self, model, n_drop, indices, num_labels):
        model.train()

        probs = torch.zeros([len(indices), num_labels]).to(self.device)

        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )

        # Dropout approximation for output probs.
        for _ in range(n_drop):
            index = 0
            for batch in iter:
                x, lengths = batch.text
                probs_i = model.predict_probs(x, lengths=lengths)
                start = index
                end = start + x.shape[0]
                probs[start:end] += probs_i
                index = end

        probs /= n_drop

        return probs

    def _get_grad_embedding(
        self,
        model,
        criterion,
        indices,
        num_targets,
        grad_embedding_type="bias_linear",
    ):

        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )

        encoder_dim = model.get_encoder_dim()
        
        # Create the tensor to return depending on the grad_embedding_type, which can have bias only,
        # linear only, or bias and linear.
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(indices), num_targets]).to(self.device)
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(indices), encoder_dim * num_targets]).to(
                self.device
            )
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros(
                [len(indices), (encoder_dim + 1) * num_targets]
            ).to(self.device)
        else:
            raise ValueError(
                f"Grad embedding type '{grad_embedding_type}' not supported."
                "Viable options: 'bias', 'linear', or 'bias_linear'"
            )

        index = 0

        for batch in iter:
            x, lengths = batch.text
            start = index
            end = start + x.shape[0]

            out, return_dict = model(x, lengths=lengths, freeze=True)
            l1 = return_dict["hidden"]
            if num_targets == 1:
                y_pred = torch.sigmoid(out)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1).to(self.device)

            y_pred = out.max(1)[1]

            # Calculate loss as a sum, allowing for the calculation of
            # the gradients using autograd wrt the outputs (bias gradients).
            loss = criterion(out, y_pred, reduction="sum")
            l0_grads = torch.autograd.grad(loss, out)[0]

            # Calculate the linear layer gradients as well if needed.
            if grad_embedding_type != "bias":
                l0_expand = torch.repeat_interleave(l0_grads, hidden_dim, dim=1)
                l1_grads = l0_expand * l1.repeat(1, num_targets)

            # Populate embedding tensor according to the supplied argument.
            if grad_embedding_type == "bias":
                grad_embedding[start:end] = l0_grads
            elif grad_embedding_type == "linear":
                grad_embedding[start:end] = l1_grads
            else:
                grad_embedding[start:end] = torch.cat([l0_grads, l1_grads], dim=1)

            index = end

            # Empty the cache as the gradient embeddings could be very large.
            torch.cuda.empty_cache()

        return grad_embedding


class RandomSampler(Sampler):
    name = "random"

    def query(self, query_size, unlab_inds, **kwargs):
        return np.random.choice(unlab_inds, size=query_size, replace=False)
