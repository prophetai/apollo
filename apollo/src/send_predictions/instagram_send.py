import matplotlib.pyplot as plt
import pandas as pd
from pandas.table.plotting import table # EDIT: see deprecation warnings below

def make_image(df):
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, df)  # where df is your data frame

    plt.savefig('instagram_picture.png')

def create_photo_container()

def post_to_instagram(user_id, creation_id):
    URL = f'graph.facebook.com/{user_id}/media_publish?creation_id={creation_id}'


    