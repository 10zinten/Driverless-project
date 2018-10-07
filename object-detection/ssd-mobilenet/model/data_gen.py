'Tran and validation data generator'

class TrainingData:

    def __init__(self, data_dir, param):
        train_samples = None
        val_samples = None

        self.preset = None
        self.num_class = None
        self.train_tfs = None
        self.val_tfs = None
        self.train_generator  = self.__build_generator(train_samples, self.train_tfs)
        self.val_generator    = self.__build_generator(val_samples, self.val_tfs)
        self.num_train = None
        self.num_val = None
        self.train_samples = None
        self.val_samples = None

    def __build_generator(self, samples, transforms):

        def run_transforms():
            pass

        def process_samples():
            pass

        def gen_batch():
            pass

        return gen_batch


if __name__ == "__main__":
    td = TrainingData(None, None)
