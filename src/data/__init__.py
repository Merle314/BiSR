from importlib import import_module
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            if args.multiScale:
                m = import_module('data.' + 'ms'+args.data_train.lower())
                dataset = getattr(m, args.data_train)(args)
            else:
                m = import_module('data.' + args.data_train.lower())
                dataset = getattr(m, args.data_train)(args)

            self.loader_train = dataloader.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K']:
                if args.multiScale:
                    m = import_module('data.msbenchmark')
                    testset = getattr(m, 'MSBenchmark')(args, train=False, name=d)
                else:
                    m = import_module('data.benchmark')
                    testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=1,
                )
            )
