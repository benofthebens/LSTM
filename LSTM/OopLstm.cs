using CsvHelper;
using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace LSTM
{


    class Weights
    {
        public double Uweight { get; set; }
        public double Wweight { get; set; }
        public double bias { get; set; }

        public Weights WeightUpdate(Weights Originalweights, double LearningRate)
        {
            Originalweights.Uweight -= Uweight * LearningRate;
            Originalweights.Wweight -= Wweight * LearningRate;
            Originalweights.bias -= bias * LearningRate;
            return Originalweights;
        }
    }
    class LstmModel
    {
        public static List<LstmLayer> LayerList = new List<LstmLayer>();
        public static int[] input;
        

        public LstmModel(int[] _input)
        {
            input = _input;
            LstmLayer.SetWeights();

        }
        public double forward()
        {

            double longMemory = 0;
            double shortMemory = 0;
            for (int i = 0; i < input.Length; i++)
            {
                (longMemory, shortMemory) = LstmUnit(input[i], longMemory, shortMemory);

            }
            return shortMemory;
        }
        public double train(int[] Expected)
        {
            double learningRate = 0.01;
            double shortMemory = forward();
            for (int i = 0; i < input.Length; i++)
            {
                UpdateWeights(input[i], Expected[i], LayerList[i], learningRate);
            }

            return shortMemory;
        }
        public void UpdateWeights(double input, double Expected, LstmLayer layer, double learningRate)
        {
            List<Weights> weights = new List<Weights>
            {
                layer.forgetGate.BackStep(input, Expected, layer).WeightUpdate(layer.forgetGate.weights, learningRate),
                layer.inputGate.BackStep(input, Expected, layer).WeightUpdate(layer.inputGate.weights,learningRate),
                layer.canidateGate.BackStep(input, Expected, layer).WeightUpdate(layer.canidateGate.weights, learningRate),
                layer.outputGate.BackStep(input, Expected, layer).WeightUpdate(layer.outputGate.weights, learningRate)

            };

            using (var writer = new StreamWriter("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\newCsv.csv",false))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                
              csv.WriteRecords(weights);
                

            }
        }
        public (double, double) LstmUnit(double input, double longterm, double shortTerm)
        {
            LstmLayer layer = new LstmLayer(input, shortTerm);

            layer.PrevCellState = longterm;
            layer.PrevHiddenState = shortTerm;
            //calculates the percentage of loing term mermory to remeber

            double longRemeberPercent = Sigmoid.sigmoid(layer.forgetGate.WeightedSum);
            layer.forgetGate.output = longRemeberPercent;
            //-----------------------------------------------------------------------------------
            //creates a new potential long term memory and determines what percentage to remeber
            double potentialRemeberPercent = Sigmoid.sigmoid(layer.inputGate.WeightedSum);
            layer.inputGate.output = potentialRemeberPercent;

            double potentialMemory = Math.Tanh(layer.canidateGate.WeightedSum);
            layer.canidateGate.output = potentialMemory;
            //------------------------------------------------------------------------------------
            //update the longtermMemory
            double UpdatedLongMemory = (longterm * longRemeberPercent) + (potentialRemeberPercent * potentialMemory);
            //------------------------------------------------------------------------------------
            //create a new short term memory and determine what percentage to remember
            double OutputPercent = Sigmoid.sigmoid(layer.outputGate.WeightedSum);
            double UpdatedShortTermMemory = Math.Tanh(UpdatedLongMemory) * OutputPercent;

            layer.HiddenState = UpdatedShortTermMemory;
            layer.outputGate.output = UpdatedShortTermMemory;

            layer.CellState = UpdatedLongMemory;
            LayerList.Add(layer);
            //return updated long term and short term memory
            return (UpdatedLongMemory, UpdatedShortTermMemory);

        }
    }
    class LstmLayer
    {
        //The gates
        //----------------------------------
        public ForgetGate forgetGate = new ForgetGate();
        public InputGate inputGate = new InputGate();
        public CanidateGate canidateGate = new CanidateGate();
        public OutputGate outputGate = new OutputGate();
        //-----------------------------------
        //Attributes of the layer
        //-----------------------------------
        public double CellState;
        public double HiddenState;
        public double PrevHiddenState;
        public double PrevCellState;
        //---------------------------------
        public static List<Weights> weightList = new List<Weights>();
        public LstmLayer(double input, double shortTerm)
        {
            forgetGate.weights = weightList[0];
            inputGate.weights = weightList[1];
            canidateGate.weights = weightList[2];
            outputGate.weights = weightList[3];

            forgetGate.WeightedSum = (shortTerm * forgetGate.weights.Uweight) + (input * forgetGate.weights.Wweight) + forgetGate.weights.bias;
            inputGate.WeightedSum = (shortTerm * inputGate.weights.Uweight) + (input * inputGate.weights.Wweight) + inputGate.weights.bias;
            canidateGate.WeightedSum = (shortTerm * canidateGate.weights.Uweight) + (input * canidateGate.weights.Wweight) + canidateGate.weights.bias;
            outputGate.WeightedSum = (shortTerm * outputGate.weights.Uweight) + (input * outputGate.weights.Wweight) + outputGate.weights.bias;
        }

        public static void SetWeights()
        {
            using (var reader = new StreamReader("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\newCsv.csv"))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                var records = csv.GetRecords<Weights>().ToList();



                foreach (var record in records)
                {
                    Weights weights = new Weights();
                    weights.Uweight = record.Uweight;
                    weights.Wweight = record.Wweight;
                    weights.bias = record.bias;

                    weightList.Add(weights);
                }


            }


        }


    }

    //sigmoid activation function 
    class Sigmoid
    {
        public static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public static double Derivative(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));

        }
    }
    //tanh activation function
    class Tanh
    {
        public static double Derivative(double x)
        {
            double tanhX = Math.Tanh(x);
            return 1 - Math.Pow(tanhX, 2);
        }

    }
    class Error
    {
        public static double Derivative(double output, double Expected)
        {
            return (output - Expected);
        }
    }
    abstract class Gate
    {
        public Weights weights { get; set; }

        public double WeightedSum { get; set; }
        public double output;
        public abstract Weights BackStep(double input, double Expected, LstmLayer layer);



    }
    class ForgetGate : Gate
    {
        public override Weights BackStep(double input, double Expected, LstmLayer layer)
        {
            Weights weights = new Weights();
            double de_df = Error.Derivative(Expected, layer.HiddenState) * layer.HiddenState * Sigmoid.Derivative(layer.forgetGate.WeightedSum) * Tanh.Derivative(layer.CellState) * layer.PrevCellState;
            weights.Uweight = de_df * input;
            weights.Wweight = de_df * layer.PrevHiddenState;
            weights.bias = de_df;
            return weights;

        }
    }
    class InputGate : Gate
    {

        public override Weights BackStep(double input, double Expected, LstmLayer layer)
        {
            Weights weights = new Weights();
            double de_di = Error.Derivative(Expected, layer.outputGate.output) * layer.canidateGate.output * Sigmoid.Derivative(layer.inputGate.WeightedSum) * layer.outputGate.output * Tanh.Derivative(layer.CellState);
            weights.Uweight = de_di * input;
            weights.Wweight = layer.PrevHiddenState * de_di;
            weights.bias = de_di;
            return weights;

        }
    }
    class CanidateGate : Gate
    {
        public override Weights BackStep(double input, double Expected, LstmLayer layer)
        {
            Weights weights = new Weights();
            double de_dg = Error.Derivative(Expected, layer.outputGate.output) * Tanh.Derivative(layer.CellState) * layer.outputGate.output * layer.inputGate.output * Sigmoid.Derivative(layer.canidateGate.WeightedSum);
            weights.Uweight = de_dg * input;
            weights.Wweight = de_dg * layer.PrevHiddenState;
            weights.bias = de_dg;
            return weights;
        }

    }
    class OutputGate : Gate
    {
        public override Weights BackStep(double input, double Expected, LstmLayer layer)
        {
            Weights weights = new Weights();
            double de_do = Error.Derivative(Expected, layer.HiddenState) * Sigmoid.Derivative(layer.outputGate.WeightedSum) * Math.Tanh(layer.CellState);
            weights.Uweight = de_do * input;
            weights.Wweight = de_do * layer.PrevHiddenState;
            weights.bias = de_do;
            return weights;
        }

    }
}