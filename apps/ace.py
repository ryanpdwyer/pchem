import streamlit as st
from streamlit_ace import st_ace


# See 
# https://discuss.streamlit.io/t/take-code-input-from-user/6413/2?u=ryanpdwyer

# import random, string
# import importlib
# import os

# strategy_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) 
# with open(strategy_name+'.py', 'w') as the_file:
#     the_file.write(content)
# TestStrategy = getattr(importlib.import_module(strategy_name), 'TestStrategy')

# # do stuff
# if os.path.exists(strategy_name+'.py'):
#   os.remove(strategy_name+'.py')
# else:
#   print("The file does not exist")


def run():
    data = st.file_uploader("Upload data files:", accept_multiple_files=True)

    # Spawn a new Ace editor
    content = st_ace(language='python')
    # Display editor's content as you type
    st.write(content)

    # I should probably persist the code in some way?
    # streamlit generates code, user tweaks for their application?

if __name__ == '__main__':
    run()