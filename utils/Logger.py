import logging

serlogger = logging.getLogger('ser_logger')
serlogger.setLevel(logging.INFO)
file_hander = logging.FileHandler('./log/ser_log.log')
file_hander.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_hander.setFormatter(formatter)
serlogger.addHandler(file_hander)

cltlogger = logging.getLogger('clt_logger')
cltlogger.setLevel(logging.INFO)
clt_file_hander = logging.FileHandler('./log/clt_log.log')
clt_file_hander.setLevel(logging.INFO)
clt_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
clt_file_hander.setFormatter(clt_formatter)
cltlogger.addHandler(clt_file_hander)

# cltloggers = [None for _ in range(10)]
# file_handers = [None for _ in range(10)]
# for i in range(10):
#     cltloggers[i] = logging.getLogger(f'clt_logger{i}')
#     cltloggers[i].setLevel(logging.INFO)
#     file_handers[i] = logging.FileHandler(f'./log/clt{i}_log.log')
#     file_handers[i].setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handers[i].setFormatter(formatter)
#     cltloggers[i].addHandler(file_handers[i])