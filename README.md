This shit runs on python 3.6 in the `gptbot_service.py` file.
It is hard coded to use a specific server.
idk if you can dockerize this.
This relies on trained set data that is huge and stored on google drive.
Each set should be in its own folder under the "checkpoint" folder at this level.

DEPENDENCIES
------------
The DISCORD_listen_generate.py requires:
- python 3.6 (do this with conda) `conda create -n gpt_bot_env python=3.6`
    - discord.py 1.7.3            `pip install discord.py==1.7.3`
    - gpt_2_simple 0.7            `gpt_2_simple==0.7`
    - tensorflow 1.14.0           `pip install tensorflow==1.14`
    - googletrans                 `pip install googletrans`
    - pillow                      `pip install pillow`
    - faker                       `pip install faker`
    - (if you want to train) cuda v10.0
