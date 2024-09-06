import logging
import os
 
def create_logger(logfile):
    """
    Creates log file for different api calls
    """
    print('Inside create_logger function')
    log_file_path=logfile
    if not os.path.exists(log_file_path):
        print('File doesnt exist')
        with open(log_file_path,"w") as f:
            f.write("Begin Logs\n")
    else:
        print('skip')
        pass
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename = f"{logfile}",
                    format = '%(asctime)s %(message)s',
                    filemode = 'a')
    logger1 = logging.getLogger()
    logger1.setLevel(logging.INFO)
    return logger1