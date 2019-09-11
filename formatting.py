import datetime

# Base, helper and formatting functions:
class bcolors:
    CYAN = '\033[36m'
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    RED = '\033[91m'
    WHITE = '\033[97m'

    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    BGBLUE = '\033[44m'
    BGRED = '\033[41m'
    BGYELLOW = '\033[43m'
    BGCYAN = '\033[46m'
    BGBLACK = '\033[40m'
    BGLGRAY = '\033[47m'
    BGGRAY = '\033[100m'

    ENDC = '\033[0m'

def now_string():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def standard_msg(message, nl=False):
    if nl:
        print("\n" + now_string() + ": %s" % message)
    else:
        print(now_string() + ": %s" % message)

def vimp_msg(message, nl=False):
    if nl:
        print("\n" + bcolors.WHITE + bcolors.BGGRAY + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)
    else:
        print(bcolors.WHITE + bcolors.BGGRAY + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)

def imp_msg(message, nl=False):
    if nl:
        print("\n" + bcolors.WHITE + bcolors.BGLGRAY + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)
    else:
        print(bcolors.WHITE + bcolors.BGLGRAY + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)

def warn_msg(message, nl=False):
    if nl:
        print("\n" + bcolors.WHITE + bcolors.BGRED + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)
    else:
        print(bcolors.WHITE + bcolors.BGRED + bcolors.BOLD + now_string() + ": %s" % message + bcolors.ENDC)
