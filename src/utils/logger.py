import time

class Logger(object):

    def __init__(self, path, show=True):
        super(Logger, self).__init__()
        self.path = path
        self.show = show
        self.log_file = open(path, 'w', encoding='utf-8')
        self.start = time.time()

    def log(self, text: str) -> None:
        end = time.time()
        text = text + ('\t%.4f' % (end - self.start))
        if self.show:
            print(text)
        self.log_file.write(text + '\n')

    def log_dict(self, d: dict):
        for key, value in d.items():
            self.log('%s: %s' % (str(key), str(value)))

    def __del__(self):
        self.log_file.close()