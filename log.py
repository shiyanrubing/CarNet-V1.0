import logging
import os
from crack_train import parse_args

def get_logger(f='log.txt', mode='w'):
    logger = logging.getLogger('CarNet')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(funcName)s - %(lineno)d: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # fh = logging.FileHandler(f, mode=mode)
    args = parse_args()
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    fh = logging.FileHandler(os.path.join(args.param_dir, f), mode=mode)

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':
    logger = get_logger()
    logger.info('test')
