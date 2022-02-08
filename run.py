from json import load
from al.sampler_mapping import get_al_sampler
from al.active_learner import ActiveLearner
from dataloaders import *
from train import *

import pickle
import logging
from datetime import datetime


from util import set_seed_everywhere


if __name__ == "__main__":
    args = make_parser()
    seeds = list(range(1, args.repeat + 1))

    dataloader = dataset_loaders[args.data]

    meta = Config()

    (train, val, test), vocab = dataloader()
    meta.vocab = vocab
    meta.num_tokens = len(vocab)
    meta.padding_idx = vocab.get_padding_index()
    meta.num_labels = len(train.field("label").vocab)

    # Construct correlation metrics
    correlations = [get_corr(key)() for key in args.correlation_measures]

    for sampler_name in args.al_samplers:
        result_list = []

        # Initialize logging
        fmt = "%Y-%m-%d-%H-%M"
        start_time = fname = datetime.now().strftime(fmt)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    f"log/{args.data}-{args.model_name}-{sampler_name}-{start_time}.log"
                ),
                logging.StreamHandler(),
            ],
        )

        meta_info = {
            "dataset": args.data,
            "model": args.model_name,
            "al_sampler": sampler_name,
            "warm_start_size": args.warm_start_size,
            "query_size": args.query_size,
            "batch_size": args.batch_size,
            "epochs_per_train": args.epochs,
            "seeds": seeds,
            "interpreters": args.interpreters,
        }
        logging.info(meta_info)

        for i, seed in zip(range(1, args.repeat + 1), seeds):
            logging.info(f"Running experiment {i}/{args.repeat}")
            logging.info(f"=" * 100)

            set_seed_everywhere(seed)
            logging.info(f"Seed = {seed}")
            logging.info(f"Maximum train size: {len(train)}")

            cuda = torch.cuda.is_available() and args.gpu != -1
            device = (
                torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")
            )

            # Setup the loss fn
            if meta.num_labels == 2:
                # Binary classification
                criterion = nn.BCEWithLogitsLoss()
                meta.num_targets = 1
            else:
                # Multiclass classification
                criterion = nn.CrossEntropyLoss()
                meta.num_targets = meta.num_labels

            sampler_cls = get_al_sampler(sampler_name)
            sampler = sampler_cls(
                dataset=train,
                batch_size=args.batch_size,
                device=device,
            )
            active_learner = ActiveLearner(sampler, train, val, device, args, meta)

            results = active_learner.al_loop(
                create_model_fn=initialize_model,
                criterion=criterion,
                warm_start_size=args.warm_start_size,
                query_size=args.query_size,
                correlations=correlations,
            )
            result_list.append(results)

        fname = f"{args.data}-{args.model_name}-{sampler.name}-all-{start_time}.pkl"
        with open(f"results/{fname}", "wb") as f:
            pickle.dump((result_list, meta_info), f)
