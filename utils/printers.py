# Helper to print in terminal with colors

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def my_print(*args, add_sep=False, color=bcolors.WARNING):
    # print with orange
    text = " ".join(args)
    if add_sep:
        text = "-"*50+"\n"+text+"\n"+"-"*50
    print(color, text, bcolors.ENDC)
