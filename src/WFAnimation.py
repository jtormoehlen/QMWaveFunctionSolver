import os
import imageio as iio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation


def render_frame(x_label='$x$', y_label='Re{$\Psi(x,0)$}', x_limit=[-2, 2], y_limit=[-5, 5], show=False):
    plt.xlabel(r'' + x_label + '')
    plt.ylabel(r'' + y_label + '')
    plt.xlim(x_limit[0], x_limit[1])
    plt.ylim(y_limit[0], y_limit[1])
    # plt.gca().set_aspect('equal')
    plt.show() if show else 0


def render_anim(location, delay=50):
    fig = plt.gcf()
    frames = []
    dpath = 'img/temporary'
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    for i in range(0, 100, 1):
        fpath = dpath + '/' + location + str(i) + '.png'
        img = mpimg.imread(fpath)
        frame = plt.imshow(img, animated=True)
        plt.gca().axis('off')
        frames.append([frame])
        os.remove(fpath)
    animation.ArtistAnimation(fig, frames, interval=delay, blit=True)
    plt.show()


def save_anim(location):
    dpath = 'img/dynamic'
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    with iio.get_writer(dpath + '/' + location + '.gif', mode='I') as writer:
        for i in range(0, 100, 1):
            fpath = 'img/temporary/' + location + str(i) + '.png'
            image = iio.imread(fpath)
            writer.append_data(image)
            os.remove(fpath)
    print('Saving ' + os.getcwd() + '/img/dynamic/' + str(location) + '.gif')


def save_frame(location):
    ftype = location
    dpath = 'img/temporary'
    location = 'img/temporary/' + location
    os.makedirs(dpath) if not os.path.isdir(dpath) else 0
    for i in range(0, 100, 1):
        fpath = location + str(i) + '.png'
        if not os.path.exists(fpath):
            plt.savefig(fpath)
            progress = round(((i + 1) / 100.) * 100.)
            if progress % 10 == 0:
                print('Loading ' + ftype + ': ' + str(progress) + '%')
            break
    plt.cla()
