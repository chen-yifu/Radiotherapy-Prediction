# Print with color
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKCYAN = '\033[96m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_with_color(*args, color=None):
    if color is None:
        print(*args)
    else:
        print(color, end="")
        print(*args, end="")
        print(bcolors.ENDC)