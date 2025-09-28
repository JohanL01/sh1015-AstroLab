# sh1015-AstroLab

NO TEACHERS ALLOWED 

A Guide for future Physics engineeringstudents at KTH with little python knowledge to get a headstart/insperation for possible solutions before enterring the lab session. This is part of google, and in the lab description googling is a big part of being an engineer. So do not feel like your going against the rules by looking at this repo

If you are an offended course examinator please contact me at johliu@kth.se

Guide:

If you need help to download the repository, you will have to get help from another guide (or ask Chatgpt). This guide expects you to have cloned the repo.

To start get the code up and running we have to first create a virtual enviroment. Note: If you have set the python path to python3 (You type: python3 xxx.py to run python programs, ofc use this in the command)

python -m venv .venv

This command creates a local "clean" python with no packages installed. Why do this you might ask? Because otherwise you dump the packages into your global python library. Super controversial, and you never know what packages are in each users library. 

Next step is to activate this enviroment. Here it is different depending on what OS you are using. Python has created scripts for us to use inside this ".venv" folder to activate the enviroment. 

For windows powershell:
.venv/Scrips/activate

For linux (mac):
source .venv/bin/activate

Now all python commands are run using this local python. 

Next we have to download the packages I used in my code. This is done by reading the "requirements.txt" file I have created for you. This command runs pip, the normal python package downloader, by reading '-r' the packages to download from a file 'requirements.txt'. If you want to use a new package you just download it by typing "pip name_of_your_package".

pip -r "requirements.txt"

Now try to run the main file (make sure there is a '(.venv)' text in the terminal ensuring that the virtual enviroment is activated)

python sh1015.py