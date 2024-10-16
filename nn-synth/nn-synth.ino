

#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#define _2_OPTIMIZE 0B00000000 // Enable 0B01.. for NO_BIAS or 0B001.. for MULTIPLE_BIASES_PER_LAYER
#define _1_OPTIMIZE 0B00010000 // https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
#define Tanh                   // Comment this line to use Sigmoid (the default) activation function

#include <NeuralNetwork.h>

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


const unsigned int layers[] = {3, 5, 3}; // 3 layers (1st)layer with 3 input neurons (2nd)layer 5 hidden neurons each and (3rd)layer with 1 output neuron
float *output; // 3rd layer's output(s)

//Default Inputs/Training-Data
const float inputs[3][1] = {
  {0}, // = 0
  {0}, // = 1
  {0}, // = 1
};

const float expectedOutput[2][3] = {{0.2,0.4,0.9}, {0.3,0.6,0.1}, {0.3,0.6,0.1}}; // values that we are expecting to get from the 3rd/(output)layer of Neural-Network, in other words something like a feedback to the Neural-network.
NeuralNetwork NN(layers, NumberOf(layers)); // Creating a Neural-Network with default learning-rates

void setup()
{
  AudioMemory(10);

  // Comment these out if not using the audio adaptor board.
  // This may wait forever if the SDA & SCL pins lack
  // pullup resistors
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.2);
  // sgtl5000_1.inputSelect(AUDIO_INPUT_MIC);
  
  waveform1.begin(WAVEFORM_SAWTOOTH);
  waveform1.frequency(200);
  waveform1.amplitude(0.5);
  waveform2.begin(WAVEFORM_SAWTOOTH);
  waveform2.frequency(205);
  waveform2.amplitude(0.5);
  waveform3.begin(WAVEFORM_SAWTOOTH);
  waveform3.frequency(210);
  waveform3.amplitude(0.5);

  Serial.begin(9600);




  // do{ 
  //   for (unsigned int j = 0; j < NumberOf(inputs); j++) // Epoch
  //   {
  //     NN.FeedForward(inputs[j]);      // FeedForwards the input arrays through the NN | stores the output array internally
  //     NN.BackProp(expectedOutput[j]); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
  //   }
    
  //   // Prints the Error.
  //   Serial.print("MSE: "); 
  //   Serial.println(NN.MeanSqrdError,6);

  //   // Loops through each epoch Until MSE goes < 0.003
  // }while(NN.getMeanSqrdError(NumberOf(inputs)) > 0.003);


  Serial.println("\n =-[OUTPUTS]-=");


  //Goes through all the input arrays
  for (unsigned int i = 0; i < NumberOf(inputs); i++)
  {
    output = NN.FeedForward(inputs[i]); // FeedForwards the input[i]-array through the NN | returns the predicted output(s)
    Serial.println(output[0], 7);       // Prints the first 7 digits after the comma.
  }
  // NN.print();                           // Prints the weights and biases of each layer
}
int count=0;

void loop() {
  
  output = NN.FeedForward(inputs[++count % 2]);

  waveform1.frequency(map(output[0], 0, 1, 200, 500));
  waveform2.frequency(map(output[1], 0, 1, 200, 500));
  waveform3.frequency(map(output[2], 0, 1, 200, 500));
  delay(500);

}
