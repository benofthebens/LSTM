using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM
{
    public class Question
    {
        public static int QuestionID;
        public string QuestionName;
        public int questionOutcome;
        public double propabilltyOfGettingCorrect = 0;
        public int Answer;
        public List<int> ResultList = new List<int>();
        

       


        

        public Question(string _QuestionName, int _Answer) 
        {
            QuestionID++;
            QuestionName = _QuestionName;
            Answer = _Answer;
            
        }
        public void askQuestion(int answer1)
        {

            if (answer1 == Answer)
            {
                questionOutcome = 1;

            }
            else
            {
                questionOutcome = 0;
            }
            ResultList.Add(questionOutcome);
        }

    }
}
