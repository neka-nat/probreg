import logging

log = logging.getLogger("probreg")
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
log.addHandler(ch)
