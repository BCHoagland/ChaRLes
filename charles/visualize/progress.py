import sys
from termcolor import colored

emojis = ['\N{winking face}', '\U0001F923', '\U0001F643', '\U0001F989', '\U0001F995', '\U0001F996', '\U0001F433', '\U0001F30E', '\U00002604', '\U0001F495']
# phases of moon

def progress(i, total, action, use_emoji=False):
    i = i + 1
    ratio = i / total
    percent = 100 * ratio
    filled = int(round(20 * ratio))

    if use_emoji:
        emoji = emojis[hash(action) % len(emojis)]
        bar = (emoji + ' ') * filled + '- ' * (20 - filled)
    else:
        bar = '█' * filled + '-' * (20 - filled)

    if percent <= 40:
        color = 'red'
    elif percent < 100:
        color = 'yellow'
    else:
        color = 'green'

    sys.stdout.write('\r%s |%s| ' % (action.ljust(30), bar) + colored('{0:.0f}%'.format(percent), color))

    if i == total:
        sys.stdout.write('\n')

    sys.stdout.flush()
