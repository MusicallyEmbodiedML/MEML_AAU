//INSTALL https://github.com/rlogiacco/CircularBuffer from the library manager

#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#define _2_OPTIMIZE 0B00000000 // Enable 0B01.. for NO_BIAS or 0B001.. for MULTIPLE_BIASES_PER_LAYER
#define _1_OPTIMIZE 0B00010000 // https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
#define Tanh                   // Comment this line to use Sigmoid (the default) activation function

#include <NeuralNetwork.h>
#include <vector>
#include <CircularBuffer.hpp>

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>



// GUItool: begin automatically generated code
AudioSynthWaveform       waveform1;      //xy=401,161
AudioSynthWaveform       waveform2;      //xy=409,214
AudioSynthWaveform       waveform3; //xy=417,279
AudioMixer4              mixer1;         //xy=704,199
AudioOutputI2S           i2s1;           //xy=1006,212
AudioConnection          patchCord1(waveform1, 0, mixer1, 0);
AudioConnection          patchCord2(waveform2, 0, mixer1, 1);
AudioConnection          patchCord3(waveform3, 0, mixer1, 2);
AudioConnection          patchCord4(mixer1, 0, i2s1, 0);
AudioConnection          patchCord5(mixer1, 0, i2s1, 1);
AudioControlSGTL5000     sgtl5000_1;     //xy=647,531
// GUItool: end automatically generated code

const int sliderPin= 14;

const size_t nInputs=1;
const size_t nOutputs=1;
const size_t patternElements=20;
const size_t patternSize = patternElements * nInputs;

const unsigned int layers[] = {patternSize, 10, 10, nOutputs}; // 3 layers (1st)layer with 3 input neurons (2nd)layer 5 hidden neurons each and (3rd)layer with 1 output neuron
float *output; 


std::vector<std::vector<float> > trainingInputs;
std::vector<std::vector<float> > trainingOutputs;

CircularBuffer<float, patternSize> patternBuffer;

enum NNMODES {TRAINING, INFERENCE};

NNMODES nnMode = NNMODES::TRAINING;


std::vector<std::vector<float>> expectedOutput {{0}, {0.5}, {1}}; 


NeuralNetwork NN(layers, NumberOf(layers)); // Creating a Neural-Network with default learning-rates

void addTrainingPoint(std::vector<float> x, size_t y) { //training inputs, and index to a set of outputs defined in expectedOutputs
  if (x.size() == patternSize) {
    trainingInputs.push_back(x);
    trainingOutputs.push_back(expectedOutput[y]);
    Serial.print("Training point added: ");
    for(auto &v: x) {
      Serial.print(v);
      Serial.print("\t");
    }
    Serial.print(" Output: ");
    Serial.println(y);
  }else{
    Serial.print("The number of inputs should be: ");
    Serial.println(patternSize);
  }
}

void undoLastTraining() {
  if (trainingInputs.size() > 0) {
    trainingInputs.pop_back();
    trainingOutputs.pop_back();
    Serial.println("Removed most recent training point");
  }else{
    Serial.println("There are no training points to remove");
  }
}

void train() {
  size_t maxEpochs = 500;
  do{ 
    for (unsigned int j = 0; j < trainingInputs.size(); j++) // Epoch
    {
      NN.FeedForward(trainingInputs[j].data());      // FeedForwards the input arrays through the NN | stores the output array internally
      NN.BackProp(trainingOutputs[j].data()); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
    }
    
    // Prints the Error.
    Serial.print("MSE: "); 
    Serial.println(NN.MeanSqrdError,6);

  }while(NN.getMeanSqrdError(trainingInputs.size()) > 0.01 && maxEpochs-- > 0);
}

void resetTraining() {
  trainingInputs.clear();
  trainingOutputs.clear();
  Serial.println("Reset training data");
}

void resetModel() {
  NN = NeuralNetwork(layers, NumberOf(layers)); // Creating a Neural-Network with default learning-rates
}

void printTrainingData() {
  Serial.println("Training data:");
  for(size_t i=0; i < trainingInputs.size(); i++) {
    Serial.print(i);
    Serial.print(": ");
    for(size_t j=0; j < patternSize; j++) {
      Serial.print(trainingInputs[i][j]);
      Serial.print("\t");
    }
    Serial.print(" :: ");
    for(size_t j=0; j < nOutputs; j++) {
      Serial.print(trainingOutputs[i][j]);
      Serial.print("\t");
    }
    Serial.println("");
  }
}

  
void setup()
{
  pinMode(sliderPin, INPUT);

  if (!SD.begin() ) {
    Serial.println("Could not start SD card");
  }
  AudioMemory(200);

  // Comment these out if not using the audio adaptor board.
  // This may wait forever if the SDA & SCL pins lack
  // pullup resistors
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.9);
  // sgtl5000_1.inputSelect(AUDIO_INPUT_MIC);
  
  waveform1.begin(WAVEFORM_SAWTOOTH);
  waveform1.frequency(200);
  waveform1.amplitude(0.3);
  waveform2.begin(WAVEFORM_SAWTOOTH);
  waveform2.frequency(205);
  waveform2.amplitude(0.3);
  waveform3.begin(WAVEFORM_SAWTOOTH);
  waveform3.frequency(210);
  waveform3.amplitude(0.3);

  Serial.begin(115200);
  Serial.setTimeout(1);
}
int count=0;

std::vector<float> p(patternSize);

void loop() {
  int sliderValue = analogRead(sliderPin);
  float sliderNorm = sliderValue / 1023.0;
  //push all inputs into the pattern buffer
  patternBuffer.push(sliderNorm);
  patternBuffer.copyToArray(p.data());

  if (Serial.available()) {
    String command = Serial.readString();
    if (command != "") {
      command = command[0]; //strip out \n
      Serial.println(command);
      if (command == "t") { //train
        train();
      } 
      else if (command == "i") { //toggle inference
        if (nnMode == NNMODES::TRAINING) {
          nnMode = NNMODES::INFERENCE;
          Serial.println("Mode: Inference");
        }else {
          nnMode = NNMODES::TRAINING;
          Serial.println("Mode: Training");
        }
      }    
      else if (command == "s") { //status
        printTrainingData();
      }    
      else if (command == "r") { //reset training
        resetTraining();
      }    
      else if (command == "m") { //reset model
        resetModel();
      }    
      else if (command == "u") { //undo last data point
        undoLastTraining();        
      }    
      else if (command == "c") { //save to SD
        SD.remove("neural.net");
        bool res = NN.save("neural.net");
        Serial.print("Network saved: ");
        Serial.println(res);
        NN.print();
      }    
      else if (command == "l") { //load from SD
        bool res = NN.load("neural.net");
        Serial.print("Network loaded: ");
        Serial.println(res);
        NN.print();
      }    
      else if (isDigit(command[0])) {
        addTrainingPoint(p, command.toInt());
      }    
    }
  }
  if (nnMode == NNMODES::INFERENCE) {
    output = NN.FeedForward(p.data());
    waveform1.frequency(map(output[0], 0, 1, 100, 500));
    waveform2.frequency(map(output[1], 0, 1, 100, 500));
    waveform3.frequency(map(output[2], 0, 1, 100, 500));
    for(size_t j=0; j < nOutputs; j++) {
      Serial.print(output[j], 7);       // Prints the first 7 digits after the comma.
      Serial.print("\t");
    }
    Serial.println("");
  }
  delay(20);

}
