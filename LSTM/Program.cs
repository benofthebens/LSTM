using System.Xml.Serialization;
using System.Data.Common;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using CsvHelper;
using System.Globalization;
using System;

internal class Program
{
    private static void Main(string[] args)
    {

        var lstm = new LSTM(true) ;
        var Input = new double[]
        {
           1,1,0,0,0
        };
        
        Console.WriteLine(lstm.forward(Input));

        
        
        List<WeightModel> Weights = new List<WeightModel>
        {
            new WeightModel() {wlr1 = lstm.wlr1,wlr2 = lstm.wlr2, blr1 = lstm.blr1,wpr1 = lstm.wpr1,wpr2 = lstm.wpr2,bpr1 = lstm.bpr1,wp1=lstm.wp1,wp2 = lstm.wp2,bp1 = lstm.bp1,wo1 = lstm.wo1,wo2 = lstm.wo2,bo1 = lstm.bo1  }
           
        };

        using (var writer = new StreamWriter("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\Weights.csv"))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(Weights);
        }
    }
}
class WeightModel
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





    public static List<LSTMLayer> layers = new List<LSTMLayer>();


    


    public LSTM(bool hasWeights) 
    {
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
    {
        var rand = new Random();
        return rand.NextDouble();  

    }
    public double asignBias()
    {
        return 0;
    }
    public double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));

    }
    public double SigmoidDeriv(double x) 
    {
        return (Sigmoid(x) * (1 - Sigmoid(x)));
    }
    public double ErrorDeriv(double Expected, double output)
    {
        return(output-Expected);

    }

    public (double,double) LstmUnit(double input, double longterm,double shortTerm)
    {
        LSTMLayer layer = new LSTMLayer();

        layer.PreviousCellState = longterm;
        layer.PreviousHiddenState = shortTerm;
        //calculates the percentage of loing term mermory to remeber
        double longRemeberPercent = Sigmoid((shortTerm * this.wlr1) + (input * this.wlr2) + blr1);
        layer.WeightedSumf = (shortTerm * this.wlr1) + (input * this.wlr2) + blr1;
        //-----------------------------------------------------------------------------------
        //creates a new potential long term memory and determines what percentage to remeber
        double potentialRemeberPercent = Sigmoid((shortTerm * this.wpr1)+(input * this.wpr2) + this.bpr1);
        layer.WeightedSumi = (shortTerm * this.wpr1) + (input * this.wpr2)+ this.bpr1;
        layer.InputGate = potentialRemeberPercent;
        double potentialMemory = Math.Tanh((shortTerm * this.wp1)+(input * this.wp2) + bp1);

        layer.WeightedSumg = (shortTerm * this.wp1) + (input * this.wp2) + bp1;
        //------------------------------------------------------------------------------------

        //update the longtermMemory
        double UpdatedLongMemory = ((longterm * longRemeberPercent)+(potentialRemeberPercent * potentialMemory));
        
        //create a new short term memory and determine what percentage to remember
        double OutputPercent = Sigmoid((shortTerm *wo1)+(input * wo2) + this.bo1);
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
        
        
         (longMemory, shortMemory) = LstmUnit(input[0], longMemory, shortMemory);


         (longMemory, shortMemory) = LstmUnit(input[1], longMemory, shortMemory);


         (longMemory, shortMemory) = LstmUnit(input[2], longMemory, shortMemory);


         (longMemory, shortMemory) = LstmUnit(input[3], longMemory, shortMemory);


         (longMemory, shortMemory) = LstmUnit(input[4], longMemory, shortMemory);
         //train(input, shortMemory);

        
       
        

        return shortMemory;

    }
    public void train(double[] input,double output)
    {
        
        var Expected = new double[]
        {
            0,0,0,1,0

        };


        for (int i = 4; i >= 0; i--)
        {
            UpdateWeights(input[i], Expected[i], output, layers[i]);

            


        }
        
        

    }
    public static double TanhDerivative(double x)
    {
        double tanhX = Math.Tanh(x);
        return 1 - Math.Pow(tanhX, 2);
    }
    
    public void UpdateWeights(double input,double Expected,double output,LSTMLayer layer)
    {
        (double Uderivf, double Wderivf, double bderivf) = forgetBackstep(input, Expected, output, layer);
        wlr1 = wlr1 - (0.5 * Uderivf);
        wlr2 = wlr2 - (0.5 * Wderivf);
        blr1 = blr1 - (0.5 * bderivf);

        (double Uderivi, double Wderivi, double bderivi) = inputBackstep(input,Expected,output, layer);
        wpr1 = wpr1 - (0.5 * Uderivi);
        wpr2 = wpr2 - (0.5 * Wderivi);
        bpr1 = bpr1 - (0.5 * bderivi);

        (double Uderivg, double Wderivg, double bderivg) = CanidateStateBackStep(input, Expected, output,  layer);
        wp1 = wp1 - (0.5 * Uderivg);
        wp2 = wp2 - (0.5 * Wderivg);
        bp1 = bp1 - (0.5 * bderivg);

        (double Uderivo, double Wderivo, double bderivo) = OutputBackstep(input, Expected, output,layer);
        wo1 = wo1 - (0.5 * Uderivo);
        wo2 = wo2 - (0.5 * Wderivo);
        bo1 = bo1 - (0.5 * bderivo);
    } 
    public (double,double,double) OutputBackstep(double input, double Expected,double output,LSTMLayer layer)
    {
        double de_do = ErrorDeriv(Expected, output) * SigmoidDeriv(layer.WeightedSumo) * Math.Tanh(layer.CellState);
        double de_dU = de_do * input;
        double de_dW = de_do * layer.PreviousHiddenState;
        double de_db = de_do;

        return (de_dU, de_dW, de_db);


        
    }
    public (double,double,double) forgetBackstep(double input, double Expected, double output, LSTMLayer layer)
    {
        double de_df = ErrorDeriv(Expected,output)*layer.OutputGate  * SigmoidDeriv(layer.WeightedSumf) * TanhDerivative(layer.CellState) * layer.PreviousCellState;
        double de_dU = de_df * input;
        double de_dW = de_df * layer.PreviousHiddenState;
        double de_db = de_df;

        return(de_dU, de_dW,de_db);
    }
    public (double,double,double) inputBackstep(double input, double Expected, double output,LSTMLayer layer)
    {
        double de_di = ErrorDeriv(Expected, output) * layer.CanidateState * SigmoidDeriv(layer.WeightedSumi) * layer.OutputGate * TanhDerivative(layer.CellState);
        double de_dU = de_di * input;
        double de_dW = layer.PreviousHiddenState * de_di;
        double de_db = de_di;
        return (de_dU, de_dW, de_db);

    }
    public(double,double,double) CanidateStateBackStep(double input, double Expected, double output, LSTMLayer layer)
    {
        double de_dg = ErrorDeriv(Expected, output) * TanhDerivative(layer.CellState) * layer.OutputGate * layer.InputGate *SigmoidDeriv(layer.WeightedSumg);
        double de_dU = de_dg * input;
        double de_dw = de_dg * layer.PreviousHiddenState;
        double de_db = de_dg;
        return (de_dU, de_dw, de_db);


    }
    
}