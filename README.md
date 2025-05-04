# raspberrypi-server

Goal:

    set up a server on my raspberry pi that will listen for a request to communicate, acknowledge it and sync. 

    communicate with this program from my other computer to the raspberry pi and have it echo all communication back to the pi. 


# Current task: 
    convert all the c code over to python 
    DONE

# Idea:
Make the raspberry pi the database. Do some object detection which will be the main project, but every time I run my camera and detect models, the code pushes these images and info about them to the pi. 
    # Sends image to the pi
        # The pi uses SQL commands to insert and SELECT from the database its storing 
    # Reasoning:
        I want practice with servers and databases which is why I am connecting to the pi via server and storing data in the database using MySQL



# Project architecture:
    [ Main PC (Client) ]
    |
    |   (TCP or HTTP requests, or even socket+json)
    |
    [ Raspberry Pi (server)]
        -Runs MySQL
        -Handles requests to log/query data

# Optional add-ons:
    - Main pc can have a GUI or CLI
    -Later throw a dashboard on top of the pi

# CheckList:
    - Train a model from scratch
        - Using the wider face dataset
    - Have the raspberry pi server to send data to
        - It will then query the data into and from the MySQL database on it.
    - Create a GUI local app to run the camera
        - Tkinter or PyQt (runs as a windowed desktop app)
            - Nice because no need for html/css just python

    # Maybes:
        - website + flask

# Notes:
    google drive + gdown for cloud dataset
    for venv on another laptop: 
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    pip freeze > requirements.txt
    python -m backend.train





ACTIVE TODO:
    - Current plan: 
        - finish the 'trained' model and test some test images on it to see if it magically worked. 
        - If it fails, go back and modify the training loop to see what went wrong. 
    - Go over training and validation loop to understand how its working more
    - Add matplot data to visualize training loss. 
    - Build the training pipeline:
        - write a dataset class 
        - load annotations
        - apply transformations
        - write model
        - train model
        - save model
    - Test model
    - Loop back if errors
    - Then flask backent routes to serve the model
    - Frontend to call routes and display detection
    - Research what comes next. Probably building the model, or formatting the data more.

Progress: (total: 26 hrs)
    4.16.2025   (4hrs)
        - planned out the strucure of the project
        - Created the file structure
        - Downloaded the dataset
        - Got the echo server running 
            - (will be modified and implemented later. This was mostly for practice)
    
    4.17.2025   (3hrs)
        - Wrote the parse function for the annotation file
        - Did a lot of research on the way the annotation file should be formatted
            - tuple containing image file path and 2d array of bbox coords ("", [[]])

    4.18.2025   (1hr)
        - Finished testing the parse file function. It contains the correct numbers of jpg filenames and when I loop through all bounding boxes, they are in the file, so I'm content with this check. 

    4.28.2025 (7hr)
        - I got a new m.2 and lost a lot of my progress (about 7 hours worth), so this
            is the time accounted for that. Lost my dataset file and train file but luckily was able to get it back since I was asking chat gpt for help. 
            
    4.28.2025 (30 min)
        - Working on restoring the code still.
    
    4.28.2025 (1 hr)
        - Finished the training and validation loop for the most part

    4.29.2025 (6 hrs)
        - Worked on fixing all the code and debugging what was missed in the parse function
        - Completed the training and validation loop
            - Lots of bugs, still currently working on that.

    4.30.2025 (1.5hr)

    5.4.2025 (2 hr)
        - Working on setting up the test file so I can quickly see if my model actually trained. 
            - Something is super wrong so its not working very well
        - Parsed the test file
        - Going to have to go back to the training loop. 





