# Summary AI

**This is currently a work in progress and is not finished yet.** 
Like my physics engine I plan to finish this project by the end of the semester.

**NOTE:** I changed the project from notes to summary AI because I realised it would be very hard getting enough data to make a good AI model.

**Update(Jan 30, 2025):** I have recently done a overhall of everything by switching from tensorflow to pytorch and have used transformers instead of RNNs or Long-Short term memory model. This is due to pytorch being easier to use and transformers being better, parallel training, and faster. I have recently finsihed the code for the project. However, the model requires a lot of resources and time to train so it will be a bit longer until I have everything trained and setup for testing. Just to put this into persepctive, the model would take multiple hours to days to train a ~10m param model with ~300k data points. I will be attempting to use a 7900 xtx gpu for training if pytorch allows but, pytorch doesn't have native support for AMD gpus so I will be attempting to use ZLUDA or directml. Note I have also changed some other specifications and haven't uploaded the new code to the repo yet. I will upload the trained model to this repo here when completed with all relevant info about it.

**Update(March 22, 2025):** As mentioned in the last update I have the code finished however, training is not feasable right now with what I have so I will upload the code and highlight the data I would have used if somebody with the ability or resources wants to. I have also realized that I need to work on my fundamental programming skills and knowledge. As I move into my next stage of life, college, I plan to start bulding my programming skills from the ground up again and focusing my efforts more strategically into computer science. My next projects will involve learning about low level programming which includes assembly and C. Not only this but I wish to learn more about linux and switch to a linux based desktop which I blieve will be Arch Linux or something similar.

## About

I plan to develop an AI capable of processing PDF or text files and creating a summary of the document. The summaries will preserve the core information and context. The AI should be able to summarize large and complex documents. The AI will then output a piece of text that can be put into a .txt, .pdf file or raw text.

**Input:** .pdf file or .txt file

**Output:** .pdf file, .txt file, or raw text

**Backend:** Python 3.13

**Libraries/Frameworks:** PyTorch, transformers, tqdm, pandas

**Data:** Currently going to train off the CNN-DailyMail summary dataset (https://huggingface.co/datasets/abisee/cnn_dailymail). This however could change in the future and possibly train it off more than this. 
