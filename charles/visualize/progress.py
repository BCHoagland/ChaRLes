import sys
from time import sleep

emojis = ['\N{winking face}', '\U0001F923', '\U0001F643', '\U0001F989', '\U0001F995', '\U0001F996', '\U0001F433', '\U0001F30E', '\U00002604', '\U0001F495']
# phases of moon

borat = 'You will never get this'

def colored(text, rgb):
    return f'\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\x1b[0m'

def out(action, bar, percent, color, done=False):
    spaces = ' ' * (len(borat) + 2)
    done = f'{spaces}\n' if done is True else ''
    sys.stdout.write('\r%s |%s| ' % (action.ljust(30), bar) + colored('{0:.0f}%'.format(percent) + done, color))

def rainbow_bar(bar, offset):
    colors = [[255, 0, 0], [255, 100, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255]]
    final = ''
    for i in range(len(bar)):
        final += colored(bar[i], colors[(i // 2 + offset) % len(colors)])
    final += colored(f' {borat} ', colors[(len(bar) // 2 + offset) % len(colors)])
    return final

def get_color(ratio):
    r = int(min(max(500 * (1 - ratio), 0), 255))
    g = int(min(max(500 * ratio, 0), 255))
    return[r, g, 0]

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

    color = get_color(ratio)

    if percent == 100:
        for i in reversed(range(20)):
            out(action, rainbow_bar(bar, i), percent, color)
            sleep(0.1)

    done = (percent == 100)
    out(action, bar, percent, color, done)

    if i == total:
        sys.stdout.write('\n')

    sys.stdout.flush()
