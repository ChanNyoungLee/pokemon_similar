from act import act
import click
import numpy as np

@click.command()
@click.option('-n', '--name', default='abomasnow')
def start_model(name):
    act_instance = act(name)
    act_instance.show()

if __name__ == '__main__':
    start_model()
