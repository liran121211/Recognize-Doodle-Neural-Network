from Perceptrons_Network import *
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

class Doodle():
    def __init__(self, name, classification, data): #Constructor
        self.classification = classification
        self.name = name
        self.data = data
        self.imagesize = data.shape[1]
        self.quantity = len(data)
        self.training_data = [ [], [classification] ] #pixel X pixel , image label
        self.testing_data = [ [], [classification] ] #pixel X pixel , image label
        self.imageLibrary = [] #Contained 2D processed images

    def __str__(self): #Object Information
        return (25*'*'+'\nImages Category: {0}\nData Format: {1}x{2} Pixels\nNumber of Images: {3}\nClassification: {4}\nTraining Data Quantity: {5}\nTesting Data Quantiy: {6}\n'.format(
        self.name,  int(len(self.data[0]) ** 0.5 ) , int(len(self.data[0]) ** 0.5 ) , self.quantity, self.classification, len(self.training_data[0]), len(self.testing_data[0]) ) )

    def imageProcessor(self, image_index = 1): #Convert 1D array to 2D array
        index = 0
        len_array = int(self.imagesize ** 0.5 )
        array_2D = np.zeros(len_array ** 2).reshape(len_array, len_array)
        for row in range(len_array):
            for colum in range(len_array):
                array_2D[row, colum] = self.data[image_index][index]
                index += 1
        self.imageLibrary.append([array_2D])

    def createImageLibrary(self, DATA): # Prepare Dataset of images
        size = int(len(DATA))
        library = []
        print(Fore.BLUE + 'Processing all {0} images, Please hold.'.format(self.category))
        for index in range(size):
            library.append( self.imageProcessor(DATA[index]) )
        print(Fore.GREEN + 'Processed all {0} Images'.format(self.category))
        return library

    def splitData(self, training_split = 0.8): #Split Dataset into Training/Testing
        training_split = int(training_split * len(self.data))

        for index in range(int(len(self.data))):
            if (index < training_split):
                self.training_data[0].append(self.data[index])
                self.training_data[1].append(self.classification)

            else:
                self.testing_data[0].append(self.data[index])
                self.testing_data[1].append(self.classification)

    def showImages(self, amount = 1, speed=1000):  # Plot data with MetaPlotLib
        for image in range(amount):
            self.imageProcessor(image)

        if (type(self.imageLibrary) != list):
            print(Fore.RED + 'Image is not in sequence, Please load at least 1 image to it.!\n')
            exit()

        def animate(i):
            if (i < len(self.imageLibrary)):
                plt.cla()  # Clean last draw
                plt.imshow(self.imageLibrary[i][0], cmap='gray')  # binary = White-Black / gray = Black-White
            else:
                print('Finished....exit in 5 sec')
                time.sleep(5)
                exit()

        plt.get_current_fig_manager().window.setGeometry = (10, 10, 10, 10)
        animationFunc = FA(plt.gcf(), animate, interval=speed)
        plt.tight_layout()
        plt.show()

def joinData(firstData, secondData): #Concetanate 2 arrays as 1 larger array
    len_first = len(firstData[0])
    len_Second = len(secondData[0])
    new_length = len_first + len_Second
    new_array = [ [], [] ]
    for index in range(new_length):
        if (index < len_first):
            new_array[0].append(firstData[0][index])
            new_array[1].append(firstData[1][index])
        else:
            new_array[0].append(secondData[0][index - len_first])
            new_array[1].append(secondData[1][index - len_first])

    return new_array

def getCategory(image): #Retreive Image Classification
    if (image == 0):
        return 'Cat'
    elif (image == 1):
        return 'Cloud'
    if (image == 2):
        return 'EyeGlasses'
    else:
        return Fore.RED + 'Something went wrong, could not identify image category\n'

def Normalize(DATA): #Normalize data to float number (0.0-1.0)
    if (type(DATA) != list):
        print(Fore.RED + 'Image is not in sequence, Please put [] on the argument you sent!\n')
        exit()
    data_length = len(DATA[0])
    float_array = [[], []]
    for index in range(data_length):
        float_array[0].append(DATA[0][index] / 255.0)
        float_array[1].append(DATA[1][index])
    print(Fore.GREEN + 'Data normalization is complete.')
    return float_array

def Normalize_Single(DATA): #Normalize data to float number (0.0-1.0)
    if (type(DATA) != list):
        print(Fore.RED + 'Image is not in sequence, Please put [] on the argument you sent!\n')
        exit()
    data_length = len(DATA[0])
    float_array = [[], []]
    for index in range(data_length):
        float_array[0].append(DATA[0][index] / 255.0)
    print(Fore.GREEN + 'Data normalization is complete.')
    return float_array



#Process Data
def setup(images_to_learn = 1):
    #make Doodles objects (Global Frame) available
    global Cloud_Doodle,Cats_Doodle,EyeGlasses_Doodle

    # Load Data
    CATS_DATA = np.load('Data/Clock_Doodle.npy', encoding='latin1', allow_pickle=True)[:images_to_learn]
    CLOUD_DATA = np.load('Data/Doodle_Cloud.npy', encoding='latin1', allow_pickle=True)[:images_to_learn]
    EYEGLASSES_DATA = np.load('Data/Doodle_EyeGlasses.npy', encoding='latin1', allow_pickle=True)[:images_to_learn]

    # Initialize Doodle Objects
    Cats_Doodle = Doodle(name='Cats', classification=0, data=CATS_DATA)
    Cloud_Doodle = Doodle(name='Cloud', classification=1, data=CLOUD_DATA)
    EyeGlasses_Doodle = Doodle(name='EyeGlasses', classification=2, data=EYEGLASSES_DATA)



    # Split Data
    Cats_Doodle.splitData(training_split=0.8)
    Cloud_Doodle.splitData(training_split=0.8)
    EyeGlasses_Doodle.splitData(training_split=0.8)

    #Initialize Neural Network (Adjustable)
    NN = NeuralNetwork(784, 64, 3)

    #Prepare Data to Train (Expandable)
    TRAIN_DATA = joinData(Cats_Doodle.training_data, Cloud_Doodle.training_data)
    TRAIN_DATA = joinData(TRAIN_DATA, EyeGlasses_Doodle.training_data)


    #Prepare Data to Test Expandable
    TEST_DATA = joinData(Cats_Doodle.testing_data, Cloud_Doodle.testing_data)
    TEST_DATA = joinData(TEST_DATA, EyeGlasses_Doodle.testing_data)


    #Normalize Data
    Normalized_TRAIN_DATA = Normalize(TRAIN_DATA)
    Normalized_TEST_DATA = Normalize(TEST_DATA)


    #Train to recognize cats images
    print(Fore.BLUE + 'Neural Network began the training! ')
    length = len(Normalized_TRAIN_DATA[0])
    for index in range(length):
        randomize = rnd.randint(0,length-1)
        targets = [0, 0, 0] #only one index can be 1 as answer
        targets = np.array(targets).reshape(-1,1) #Transpose matrix
        targets[Normalized_TRAIN_DATA[1][randomize]] = 1
        NN.train(Normalized_TRAIN_DATA[0][randomize], targets)
    print(Fore.GREEN + 'Neural Network finished the training! ')

    return NN, Normalized_TEST_DATA, Normalized_TRAIN_DATA

'''def analyze(size):
    SETUP_DATA = setup(size) #load all setup data
    NN_Object = SETUP_DATA[0]
    Test_Dataset = SETUP_DATA[1]
    Train_Dataset = SETUP_DATA[2]
    correct = 0

    len_test_data = len(Test_Dataset[0]) #get quantity of images (TEST_DATA dataset)

    for image in range(len_test_data): #test all images on Test Dataset

        send_image = (Test_Dataset[0][image]) #load image to label it
        guess = np.argmax(NN_Object.prediction(send_image)) # guess float values [ x.xx, x.xx, x.xx ]

        #print(Fore.BLACK + 'Test Results: {0}'.format(NN_Object.prediction(send_image)))
        print(Fore.BLACK + 'Image sent is: {0} | Computer thinks it: {1}'.format(getCategory(Test_Dataset[1][image]), getCategory(guess)))
        if (Test_Dataset[1][image] == guess):
            correct += 1

    print(Fore.CYAN + 'Success Rate: %{0}'.format( correct/len_test_data * 100 ) )
'''
#------------Draw Doodle Section--------------
def createDoodle():
    width = 280
    height = 280

    def save():  # save the canvas image
        filename = "doodle.png"
        resize = image1.resize((28, 28), PIL.Image.ANTIALIAS)
        resize.save(filename)

    def paint(event):  # paint on the canvas
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="white", width=10)
        draw.line([x1, y1, x2, y2], fill="white", width=10)

    root = Tk()  # Create tkinter object
    root.title('Doodle Drawer')

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    # PIL create an empty image and draw object to draw on
    image1 = PIL.Image.new(mode="L", size=(height, width))  # L = GrayScale
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings (visible)
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # image save button (filename)
    button = Button(text="Save Doodle!", command=save)
    button.pack()

    root.mainloop()

    # save and convert the image to grayscale
    doodle_image = PIL.Image.open('doodle.png').convert('L')
    return doodle_image


def computerGuess(figure, neuralnetwork):


    # convert image to 1D array and normalize it
    doodle_array = np.asarray(figure).flatten()
    doodle_array_normalized = [Normalize_Single([doodle_array])]
    doodle_array_normalized = doodle_array_normalized[0][0]
    guess = np.argmax(neuralnetwork.prediction(doodle_array_normalized))
    print(neuralnetwork.prediction(doodle_array_normalized))
    print(Fore.BLACK + '\nLet me guess it was: {0} ?'.format(getCategory(guess)))



NeuralNetwork = setup(images_to_learn=20000)[0] #Learn Images first
doodle_figure = createDoodle() #create doodle of your own
computerGuess(doodle_figure, NeuralNetwork)

#Check training data of doodles
#Cats_Doodle.showImages(amount= len(Cats_Doodle.data) , speed= 500 )
