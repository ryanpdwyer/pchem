
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import requests



colormap = sns.color_palette("blend:white,green", as_cmap=True)


def run():

    response = requests.get('https://neutral-network-ex-default-rtdb.firebaseio.com/data.json')
    data = response.json()

    st.markdown("""\
# Neural network game
To play the game, build your neural networks here: [A Neural Network Playground](https://neutral-network-ex.web.app/#activation=tanh&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.75883&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
                
For each model, your points will be calculated as follows:
                
    Pts =(Max pts) - 3.5*(inputs + neurons) - (epochs)/100 - 1000*(test loss)

The goal is to make your model work as well as possible while using the fewest resources (fewest neurons, inputs, and training epochs). Press R to refresh the leaderboard.

The maximum points for each dataset are 50 for gaussian and xor, 75 for circle, and 250 for spiral.
         
You should try to get at least **200 points.** The overall winner who hasn't gotten a skip before gets out of one of our upcoming homework assignments.
                
""")
    
    d = pd.DataFrame.from_dict(data, orient='index')
    
    # Filter out datasets that are not spiral, circle, xor, or gaussian
    d = d.query('dataset == "spiral" or dataset == "circle" or dataset == "xor" or dataset == "gauss"')

    # Shorten the column names - the current names are
    # Timestamp
    # Name (include initial)
    # Which dataset are you training?
    # How many total inputs and neurons did you use? (This example would be 2 inputs + 1 neuron + 1 neuron = 4) (fewer = better)
    # How many epochs of training did you use? (this would be 161) (fewer = better)
    # What was your test loss? (here = 0.048, lower = much better)
    # Copy and paste the url to your model here

    # d = calculate_points(df)
    
    # st.write(d)


    # Remove everything after the open parenthesis in the dataset column
    # d['dataset'] = d['dataset'].str.split('(').str[0].str[:-1]

    # d['score'] = d['Max_pts'] - d['Inputs and neurons']*3.5 - d['Epochs']/100.0 - d['Test loss']*1000.0

    # best_spirals = d.groupby(['name', 'dataset']).agg('max').

    max_values = d.groupby(['name', 'dataset']).agg('max')
    max_values['score'].clip(lower=0, inplace=True)


    
    
    pts_by_category = max_values['score'].reset_index().pivot(index='name', columns='dataset', values='score')
    
    totals = max_values.groupby('name').agg('sum').sort_values('score', ascending=False)['score']

    # Add a pretty color bar to each column in pts_by_category (green for large values, red for small)
    pts_by_category['Total'] = totals
    pts_by_category.sort_values('Total', ascending=False, inplace=True)

    pts_by_category.fillna(0, inplace=True)

    # Move the total column to the front
    cols = pts_by_category.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    pts_by_category = pts_by_category[cols]


    st.markdown("""## Overall Leaderboard""")
    st.dataframe(pts_by_category.round(1).style.background_gradient(cmap=colormap, axis=0))
                 


    st.markdown("""## Leaderboard for each category""")

    # Make this selectbox default to no value
    ex = [None]
    ex.extend(pts_by_category.columns[1:].tolist())

    test_example = st.selectbox("Select a category", ex, index=0 )
    if test_example is not None:
        best_spirals = max_values.reset_index().query('`dataset` == @test_example').sort_values('score', ascending=False)
        best_spirals = best_spirals[['name', 'score', 'url']].head(5)
        st.dataframe(best_spirals.style.background_gradient(cmap=colormap, axis=0)
                     )
        
    

   

    

    


if __name__ == "__main__":
    run()
