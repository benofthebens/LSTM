using System.Xml.Serialization;
using System.Data.Common;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using CsvHelper;

using System;

internal class Failed
{
    private static void MainFaile(string[] args)
    {
        // creates a new instance of the LSTM class
        var lstm = new LSTM(true);
        //defines the input to pass
        var Input = new double[]
        {
           1,1,0,0,0
        };
        // writes the output of the forward method
        Console.WriteLine(lstm.forward(Input));


        // asigns weights to the Lstm model
        List<WeightModel> Weights = new List<WeightModel>
        {
            new WeightModel() {wlr1 = lstm.wlr1,wlr2 = lstm.wlr2, blr1 = lstm.blr1,wpr1 = lstm.wpr1,wpr2 = lstm.wpr2,bpr1 = lstm.bpr1,wp1=lstm.wp1,wp2 = lstm.wp2,bp1 = lstm.bp1,wo1 = lstm.wo1,wo2 = lstm.wo2,bo1 = lstm.bo1  }

        };
        // writes to the csv file 
        using (var writer = new StreamWriter("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\Weights.csv"))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(Weights);
        }
    }
}
class WeightModel_
{
    public double wlr1 { get; set; }
    public double wlr2 { get; set; }
    public double blr1 { get; set; }
    public double wpr1 { get; set; }
    public double wpr2 { get; set; }
    public double bpr1 { get; set; }
    public double wp1 { get; set; }
    public double wp2 { get; set; }
    public double bp1 { get; set; }
    public double wo1 { get; set; }
    public double wo2 { get; set; }
    public double bo1 { get; set; }
}
public class LSTMLayer
{
    public double WeightedSumo;
    public double WeightedSumf;
    public double WeightedSumi;
    public double WeightedSumg;

    public double CellState;
    public double PreviousCellState;
    public double OutputGate;
    public double CanidateState;
    public double PreviousHiddenState;
    public double HiddenState;
    public double InputGate;

}
class LSTM
{
    public double wlr1 { get; set; }
    public double wlr2 { get; set; }
    public double blr1 { get; set; }
    public double wpr1 { get; set; }
    public double wpr2 { get; set; }
    public double bpr1 { get; set; }
    public double wp1 { get; set; }
    public double wp2 { get; set; }
    public double bp1 { get; set; }
    public double wo1 { get; set; }
    public double wo2 { get; set; }
    public double bo1 { get; set; }





    public static List<LSTMLayerOld_> layers = new List<LSTMLayerOld_>();





    public LSTMOld(bool hasWeights)
    {// if have weights is true then read from csv but if false asign new weights and biases
        if (hasWeights)
        {


            using (var reader = new StreamReader("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\Weights.csv"))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                var records = csv.GetRecords<WeightModel>();
                foreach (var record in records)
                {
                    wlr1 = record.wlr1;
                    wlr2 = record.wlr2;
                    blr1 = record.blr1;

                    wpr1 = record.wpr1;
                    wpr2 = record.wpr2;
                    bpr1 = record.bpr1;

                    wp1 = record.wp1;
                    wpr2 = record.wlr2;
                    bp1 = record.bp1;

                    wo1 = record.wo1;
                    wo2 = record.wo2;
                    bo1 = record.bo1;

                }

            }

        }
        else
        {
            wlr1 = this.asignWeights();
            wlr2 = this.asignWeights();
            blr1 = this.asignBias();

            wpr1 = this.asignWeights();
            wpr2 = this.asignWeights();
            bpr1 = this.asignBias();

            wp1 = this.asignWeights();
            wp2 = this.asignWeights();
            bp1 = this.asignBias();

            wo1 = this.asignWeights();
            wo2 = this.asignWeights();
            bo1 = this.asignBias();

        }






    }
    public double asignWeights()
    {//asigns random weights
        var rand = new Random();
        return rand.NextDouble();

    }
    public double asignBias()
    {
        //sets the biases to 0
        return 0;
    }
    public double Sigmoid(double x)
    {
        //sigmoid function
        return 1.0 / (1.0 + Math.Exp(-x));

    }
    public double SigmoidDeriv(double x)
    {
        //sigmoid derivative function
        return (Sigmoid(x) * (1 - Sigmoid(x)));
    }
    public double ErrorDeriv(double Expected, double output)
    {
        //calculates the error derivative
        return (output - Expected);

    }

    public (double, double) LstmUnit(double input, double longterm, double shortTerm)
    {
        LSTMLayerOld_ layer = new LSTMLayerOld_();

        layer.PreviousCellState = longterm;
        layer.PreviousHiddenState = shortTerm;
        //calculates the percentage of loing term mermory to remeber
        double longRemeberPercent = Sigmoid((shortTerm * this.wlr1) + (input * this.wlr2) + blr1);
        layer.WeightedSumf = (shortTerm * this.wlr1) + (input * this.wlr2) + blr1;
        //-----------------------------------------------------------------------------------
        //creates a new potential long term memory and determines what percentage to remeber
        double potentialRemeberPercent = Sigmoid((shortTerm * this.wpr1) + (input * this.wpr2) + this.bpr1);
        layer.WeightedSumi = (shortTerm * this.wpr1) + (input * this.wpr2) + this.bpr1;
        layer.InputGate = potentialRemeberPercent;
        double potentialMemory = Math.Tanh((shortTerm * this.wp1) + (input * this.wp2) + bp1);

        layer.WeightedSumg = (shortTerm * this.wp1) + (input * this.wp2) + bp1;
        //------------------------------------------------------------------------------------

        //update the longtermMemory
        double UpdatedLongMemory = ((longterm * longRemeberPercent) + (potentialRemeberPercent * potentialMemory));

        //create a new short term memory and determine what percentage to remember
        double OutputPercent = Sigmoid((shortTerm * wo1) + (input * wo2) + this.bo1);
        double UpdatedShortTermMemory = Math.Tanh(UpdatedLongMemory) * OutputPercent;
        layer.WeightedSumo = (shortTerm * wo1) + (input * wo2) + this.bo1;
        layer.OutputGate = UpdatedShortTermMemory;

        layer.CellState = UpdatedLongMemory;
        layers.Add(layer);
        //return updated long term and short term memory
        return (UpdatedLongMemory, UpdatedShortTermMemory);

    }
    public double forward(double[] input)
    {
        double longMemory = 0;
        double shortMemory = 0;

        //manual iteration on the lstm unit to calculate the hidden state of the final layer
        (longMemory, shortMemory) = LstmUnit(input[0], longMemory, shortMemory);


        (longMemory, shortMemory) = LstmUnit(input[1], longMemory, shortMemory);


        (longMemory, shortMemory) = LstmUnit(input[2], longMemory, shortMemory);


        (longMemory, shortMemory) = LstmUnit(input[3], longMemory, shortMemory);


        (longMemory, shortMemory) = LstmUnit(input[4], longMemory, shortMemory);
        //trains the model
        //train(input, shortMemory);





        return shortMemory;

    }
    public void train(double[] input, double output)
    {
        // the expected output
        var Expected = new double[]
        {
            0,0,0,1,0

        };


        for (int i = 4; i >= 0; i--)
        {
            // updates the weights by passing the inputs and expected output and the layers
            UpdateWeights(input[i], Expected[i], output, layers[i]);




        }



    }
    public static double TanhDerivative(double x)
    {
        // the hyperbolic  tan functrion derivative
        double tanhX = Math.Tanh(x);
        return 1 - Math.Pow(tanhX, 2);
    }

    public void UpdateWeights(double input, double Expected, double output, LSTMLayerOld_ layer)
    {
        // the forget gate change weights
        (double Uderivf, double Wderivf, double bderivf) = forgetBackstep(input, Expected, output, layer);
        wlr1 = wlr1 - (0.5 * Uderivf);
        wlr2 = wlr2 - (0.5 * Wderivf);
        blr1 = blr1 - (0.5 * bderivf);
        //input gate changes weights
        (double Uderivi, double Wderivi, double bderivi) = inputBackstep(input, Expected, output, layer);
        wpr1 = wpr1 - (0.5 * Uderivi);
        wpr2 = wpr2 - (0.5 * Wderivi);
        bpr1 = bpr1 - (0.5 * bderivi);
        //canidate state change weights
        (double Uderivg, double Wderivg, double bderivg) = CanidateStateBackStep(input, Expected, output, layer);
        wp1 = wp1 - (0.5 * Uderivg);
        wp2 = wp2 - (0.5 * Wderivg);
        bp1 = bp1 - (0.5 * bderivg);
        //output gate change weights
        (double Uderivo, double Wderivo, double bderivo) = OutputBackstep(input, Expected, output, layer);
        wo1 = wo1 - (0.5 * Uderivo);
        wo2 = wo2 - (0.5 * Wderivo);
        bo1 = bo1 - (0.5 * bderivo);
    }
    public (double, double, double) OutputBackstep(double input, double Expected, double output, LSTMLayerOld_ layer)
    {
        //caluclates the error in respect to weights
        double de_do = ErrorDeriv(Expected, output) * SigmoidDeriv(layer.WeightedSumo) * Math.Tanh(layer.CellState);
        double de_dU = de_do * input;
        double de_dW = de_do * layer.PreviousHiddenState;
        double de_db = de_do;

        return (de_dU, de_dW, de_db);



    }
    public (double, double, double) forgetBackstep(double input, double Expected, double output, LSTMLayerOld_ layer)
    {
        //caluclates the error in respect to weights
        double de_df = ErrorDeriv(Expected, output) * layer.OutputGate * SigmoidDeriv(layer.WeightedSumf) * TanhDerivative(layer.CellState) * layer.PreviousCellState;
        double de_dU = de_df * input;
        double de_dW = de_df * layer.PreviousHiddenState;
        double de_db = de_df;

        return (de_dU, de_dW, de_db);
    }
    public (double, double, double) inputBackstep(double input, double Expected, double output, LSTMLayerOld_ layer)
    {
        //caluclates the error in respect to weights
        double de_di = ErrorDeriv(Expected, output) * layer.CanidateState * SigmoidDeriv(layer.WeightedSumi) * layer.OutputGate * TanhDerivative(layer.CellState);
        double de_dU = de_di * input;
        double de_dW = layer.PreviousHiddenState * de_di;
        double de_db = de_di;
        return (de_dU, de_dW, de_db);

    }
    public (double, double, double) CanidateStateBackStep(double input, double Expected, double output, LSTMLayerOld_ layer)
    {
        //caluclates the error in respect to weights
        double de_dg = ErrorDeriv(Expected, output) * TanhDerivative(layer.CellState) * layer.OutputGate * layer.InputGate * SigmoidDeriv(layer.WeightedSumg);
        double de_dU = de_dg * input;
        double de_dw = de_dg * layer.PreviousHiddenState;
        double de_db = de_dg;
        return (de_dU, de_dw, de_db);


    }

}