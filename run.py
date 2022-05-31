from json import load
from experiment import Experiment
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

    if args.model_name in TRANSFORMERS:
        tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_model)
    else:
        tokenizer = None

    (train, val, test), vocab = dataloader(meta=meta, tokenizer=tokenizer)

    # Construct correlation metrics
    correlations = [get_corr(key)() for key in args.correlation_measures]

    # Initialize logging
    fmt = "%Y-%m-%d-%H-%M"
    start_time = fname = datetime.now().strftime(fmt)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/{args.data}-{args.model_name}-{start_time}.log"),
            logging.StreamHandler(),
        ],
    )

    args.interpreters = sorted(args.interpreters)
    test_lengths = [len(ex.text[1]) for ex in test.examples]
    meta_info = {
        "dataset": args.data,
        "model": args.model_name,
        "warm_start_size": args.warm_start_size,
        "batch_size": args.batch_size,
        "epochs_per_train": args.epochs,
        "seeds": seeds,
        "interpreters": args.interpreters,
        "correlations": args.correlation_measures,
        "tying": args.tying,
        "conicity": args.conicity,
        "l2": args.l2,
    }
    logging.info(meta_info)

    result_list = []
    for i, seed in zip(range(1, args.repeat + 1), seeds):
        logging.info(f"Running experiment {i}/{args.repeat}")
        logging.info(f"=" * 100)

        set_seed_everywhere(seed)
        logging.info(f"Seed = {seed}")
        logging.info(f"Maximum train size: {len(train)}")

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

        experiment = Experiment(train, test, device, args, meta)
        meta_info["test_lengths"] = experiment.test_lengths.tolist()
        meta_info["test_mapping"] = experiment.get_id_mapping()

        results = experiment.run(
            create_model_fn=initialize_model,
            criterion=criterion,
            warm_start_size=args.warm_start_size,
            correlations=correlations,
        )
        result_list.append(results)

    fname = f"{args.data}-{args.model_name}-conicity={args.conicity}-tying={args.tying}-l2={args.l2}-{start_time}.pkl"
    with open(f"{args.save_dir}/{fname}", "wb") as f:
        pickle.dump((result_list, meta_info), f)
