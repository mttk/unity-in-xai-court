from al.active_learner import ActiveLearner
from al.uncertainty import MarginSampler
from al.sampler import RandomSampler
from datasets import *
from train import *


if __name__ == "__main__":
    args = make_parser()
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

    # Initialize model
    model = initialize_model(args, meta)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2)

    # Construct interpreters
    interpreters = {i: get_interpreter(i)(model) for i in args.interpreters}
    print(f"Interpreters: {' '.join(list(interpreters.keys()))}")

    # Construct correlation metrics
    correlations = [get_corr(key)() for key in args.correlation_measures]
    print(f"Correlation measures: {correlations}")

    sampler = RandomSampler(dataset=train, batch_size=args.batch_size, device=device)
    active_learner = ActiveLearner(sampler, train, val, device, args, meta)

    active_learner.al_loop(
        create_model_fn=initialize_model,
        optimizer=optimizer,
        criterion=criterion,
        warm_start_size=args.warm_start_size,
        query_size=args.query_size,
        interpreters=interpreters,
        correlations=correlations,
    )
