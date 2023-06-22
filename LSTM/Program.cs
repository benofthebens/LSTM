using System.Xml.Serialization;
using System.Data.Common;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using CsvHelper;
using System.Globalization;
using System;
using LSTM;

internal class Program
{
    private static void Main(string[] args)
    {
        List<Question> questionList = new List<Question>();
        Question q1 = new Question("1+1",2);
        q1.askQuestion(1);
        q1.askQuestion(1);
        q1.askQuestion(1);
        q1.askQuestion(1);
        q1.askQuestion(1);


        Question q2 = new Question("2+2", 4);
        q2.askQuestion(4);
        q2.askQuestion(4);
        q2.askQuestion(4);
        q2.askQuestion(4);
        q2.askQuestion(4);


        Question q3 = new Question("3+3", 6);
        q3.askQuestion(6);
        q3.askQuestion(7);
        q3.askQuestion(6);
        q3.askQuestion(7);
        q3.askQuestion(7);
        var lstm = new LSTMModel(true);
        var Inputs = new int[][]
        {
           new int[] { 0, 0, 0, 0, 0 },  
           new int[] { 0, 0, 0, 0, 1 },  
           new int[] { 0, 0, 0, 1, 0 },  
           new int[] { 0, 0, 0, 1, 1 },  
           new int[] { 0, 0, 1, 0, 0 },  
           new int[] { 0, 0, 1, 0, 1 },  
           new int[] { 0, 0, 1, 1, 0 },  
           new int[] { 0, 0, 1, 1, 1 },  
           new int[] { 0, 1, 0, 0, 0 },  
           new int[] { 0, 1, 0, 0, 1 },  
           new int[] { 0, 1, 0, 1, 0 },  
           new int[] { 0, 1, 0, 1, 1 },  
           new int[] { 0, 1, 1, 0, 0 },  
           new int[] { 0, 1, 1, 0, 1 },  
           new int[] { 0, 1, 1, 1, 0 },  
           new int[] { 0, 1, 1, 1, 1 },  
           new int[] { 1, 0, 0, 0, 0 },  
           new int[] { 1, 0, 0, 0, 1 },  
           new int[] { 1, 0, 0, 1, 0 },  
           new int[] { 1, 0, 0, 1, 1 },  
           new int[] { 1, 0, 1, 0, 0 },  
           new int[] { 1, 0, 1, 0, 1 },  
           new int[] { 1, 0, 1, 1, 0 },  
           new int[] { 1, 0, 1, 1, 1 },  
           new int[] { 1, 1, 0, 0, 0 },  
           new int[] { 1, 1, 0, 0, 1 },  
           new int[] { 1, 1, 0, 1, 0 },  
           new int[] { 1, 1, 0, 1, 1 },  
           new int[] { 1, 1, 1, 0, 0 },  
           new int[] { 1, 1, 1, 0, 1 },  
           new int[] { 1, 1, 1, 1, 0 },  
           new int[] { 1, 1, 1, 1, 1 }

        };

        var Expected = new int[][]
        {
            new int[] {0,0,0,0,0},
            new int[] {0,0,0,0,0},
            new int[]{0,0,0,0,1},
            new int[]{0,0,0,0,1},
            new int[]{0,0,0,1,0},
            new int[] { 0, 0, 0, 1, 0 },
            new int[] { 0, 0, 0, 1, 1 },
            new int[] { 0, 0, 0, 1, 1 },
            new int[] {0,0,1,0,0},
            new int[] { 0, 0, 1, 0, 0 },
            new int[] { 0, 1, 0, 1, 0 },
            new int[] { 0, 0, 1, 0, 1 },
            new int[] { 0, 0, 1, 0, 1 },
            new int[] { 0, 0, 1, 1, 1 },
            new int[] { 0, 0, 1, 1, 1 },
            new int[] { 0, 1, 0, 0, 0 },
            new int[] { 0, 1, 0, 0, 0 },
            new int[] { 0, 1, 0, 0, 1 },
            new int[] { 0, 1, 0, 0, 1 },
            new int[] { 0, 1, 0, 1, 0 },
            new int[] { 0, 1, 0, 1, 0 },
            new int[] { 0, 1, 0, 1, 0 },
            new int[] { 0, 1, 0, 1, 1 },
            new int[] { 0, 1, 0, 1, 1 },
            new int[] { 0, 1, 1, 0, 0 },
            new int[] { 0, 1, 1, 0, 0 },
            new int[] { 0, 1, 1, 0, 1 },
            new int[] { 0, 1, 1, 0, 1 },
            new int[] { 0, 1, 1, 1, 0 },
            new int[] { 0, 1, 1, 1, 0 },
            new int[] { 0, 1, 1, 1, 1 },
            new int[] { 0, 1, 1, 1, 1 }

        };
        /*for(int epoch = 0; epoch < 15; epoch++)
        {
           for (int i = 0; i < Expected.Length; i++)
            {

               double q1Prediction = lstm.forward(Inputs[i], Expected[i],lstm);
               Console.WriteLine(q1Prediction);
        

            }
           Console.WriteLine();
        }*/
        double q1Prediction_ = lstm.forward(q1.ResultList.ToArray(), Expected[0],lstm);
        double q2Prediction_ = lstm.forward(q2.ResultList.ToArray(), Expected[1],lstm);
        double q3Prediction_ = lstm.forward(q3.ResultList.ToArray(), Expected[2],lstm);

        

        q1.propabilltyOfGettingCorrect = q1Prediction_;
        q2.propabilltyOfGettingCorrect = q2Prediction_;
        q3.propabilltyOfGettingCorrect = q3Prediction_;
        questionList.Add(q1);
        questionList.Add(q2);
        questionList.Add(q3);
        questionList.Sort((x, y) => x.propabilltyOfGettingCorrect.CompareTo(y.propabilltyOfGettingCorrect));
        
        foreach ( var item in questionList)
        {
            Console.WriteLine(item.QuestionName);
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
class LSTMModel
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





    public LSTMModel(bool hasWeights)
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
        var rand = new Random();
        return rand.NextDouble();
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
        return (output - Expected);

    }

    public (double, double) LstmUnit(double input, double longterm, double shortTerm)
    {
        LSTMLayer layer = new LSTMLayer();

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
        layer.CanidateState = potentialMemory;
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
    public double forward(int[] input, int[]Expected, LSTMModel lstmObject)
    {
        double longMemory = 0;
        double shortMemory = 0;
        for(int i = 0; i < input.Length; i++)
        {
            (longMemory, shortMemory) = LstmUnit(input[i], longMemory, shortMemory);

        }

        


        
       //train(input,Expected,lstmObject);





        return shortMemory;

    }
    public void train(int[] input,int[]Expected,LSTMModel lstmObject)
    {

        double LearningRate = 0.01; 


        for (int i = 0; i < input.Length; i++)
        {
            UpdateWeights(input[i], Expected[i], layers[i],LearningRate);

            StoreWeights(lstmObject);


        }

        


    }
    public void StoreWeights(LSTMModel lstmObject)
    {
        List<WeightModel> Weights = new List<WeightModel>
        {
            new WeightModel() {wlr1 = lstmObject.wlr1,wlr2 = lstmObject.wlr2, blr1 = lstmObject.blr1,wpr1 = lstmObject.wpr1,wpr2 = lstmObject.wpr2,bpr1 = lstmObject.bpr1,wp1=lstmObject.wp1,wp2 = lstmObject.wp2,bp1 = lstmObject.bp1,wo1 = lstmObject.wo1,wo2 = lstmObject.wo2,bo1 = lstmObject.bo1  }

        };
        using (var writer = new StreamWriter("C:\\Users\\bjwha\\Desktop\\Code project\\LSTM\\Weights.csv"))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(Weights);
        }
    }
    public static double TanhDerivative(double x)
    {
        double tanhX = Math.Tanh(x);
        return 1 - Math.Pow(tanhX, 2);
    }

    public void UpdateWeights(double input, double Expected, LSTMLayer layer,double LearningRate)
    {
        (double Uderivf, double Wderivf, double bderivf) = forgetBackstep(input, Expected, layer);
        wlr1 = wlr1 - (LearningRate * Uderivf);
        wlr2 = wlr2 - ( LearningRate * Wderivf);
        blr1 = blr1 - ( LearningRate * bderivf);

        (double Uderivi, double Wderivi, double bderivi) = inputBackstep(input, Expected, layer);
        wpr1 = wpr1 - (LearningRate * Uderivi);
        wpr2 = wpr2 - (LearningRate * Wderivi);
        bpr1 = bpr1 - (LearningRate * bderivi);

        (double Uderivg, double Wderivg, double bderivg) = CanidateStateBackStep(input, Expected, layer);
        wp1 = wp1 - (LearningRate * Uderivg);
        wp2 = wp2 - (LearningRate * Wderivg);
        bp1 = bp1 - (LearningRate * bderivg);

        (double Uderivo, double Wderivo, double bderivo) = OutputBackstep(input, Expected, layer);
        wo1 = wo1 - (LearningRate * Uderivo);
        wo2 = wo2 - (LearningRate * Wderivo);
        bo1 = bo1 - (LearningRate * bderivo);
    }
    public (double, double, double) OutputBackstep(double input, double Expected, LSTMLayer layer)
    {
        double de_do = ErrorDeriv(Expected, layer.OutputGate) * SigmoidDeriv(layer.WeightedSumo) * Math.Tanh(layer.CellState);
        double de_dU = de_do * input;
        double de_dW = de_do * layer.PreviousHiddenState;
        double de_db = de_do;

        return (de_dU, de_dW, de_db);



    }
    public (double, double, double) forgetBackstep(double input, double Expected, LSTMLayer layer)
    {
        double de_df = ErrorDeriv(Expected, layer.OutputGate) * layer.OutputGate * SigmoidDeriv(layer.WeightedSumf) * TanhDerivative(layer.CellState) * layer.PreviousCellState;
        double de_dU = de_df * input;
        double de_dW = de_df * layer.PreviousHiddenState;
        double de_db = de_df;

        return (de_dU, de_dW, de_db);
    }
    public (double, double, double) inputBackstep(double input, double Expected, LSTMLayer  layer)
    {
        double de_di = ErrorDeriv(Expected, layer.OutputGate) * layer.CanidateState * SigmoidDeriv(layer.WeightedSumi) * layer.OutputGate * TanhDerivative(layer.CellState);
        double de_dU = de_di * input;
        double de_dW = layer.PreviousHiddenState * de_di;
        double de_db = de_di;
        return (de_dU, de_dW, de_db);

    }
    public (double, double, double) CanidateStateBackStep(double input, double Expected, LSTMLayer layer)
    {
        double de_dg = ErrorDeriv(Expected, layer.OutputGate) * TanhDerivative(layer.CellState) * layer.OutputGate * layer.InputGate * SigmoidDeriv(layer.WeightedSumg);
        double de_dU = de_dg * input;
        double de_dw = de_dg * layer.PreviousHiddenState;
        double de_db = de_dg;
        return (de_dU, de_dw, de_db);


    }

}