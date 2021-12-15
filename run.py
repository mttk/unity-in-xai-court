from al.sampler_mapping import get_al_sampler
from al.active_learner import ActiveLearner
from dataloaders import *
from train import *

import pickle
from datetime import datetime


if __name__ == "__main__":
    args = make_parser()
    dataset_name = "IMDB"
    (train, val, test), vocab = load_imdb()

    meta = Config()
    meta.num_labels = 2
    meta.num_tokens = len(vocab)
    meta.padding_idx = vocab.get_padding_index()
    meta.vocab = vocab

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")

    # Setup the loss fn
    if meta.num_labels == 2:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        meta.num_targets = 1
    else:
        # Multiclass classification
        criterion = nn.CrossEntropyLoss()
        meta.num_targets = meta.num_labels

    # Construct correlation metrics
    correlations = [get_corr(key)() for key in args.correlation_measures]
    print(f"Correlation measures: {correlations}")

    sampler_cls = get_al_sampler(args.al_sampler)
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

    print(results)
    fmt = "%Y-%m-%d-%H-%M"
    fname = f"{dataset_name}-{sampler.name}-{datetime.now().strftime(fmt)}.pkl"
    with open(f"results/{fname}.pkl", "wb") as f:
        pickle.dump(results, f)
