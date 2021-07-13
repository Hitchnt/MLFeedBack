using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLFeedBack
{
    
    class FeedBackTrainingData {

        [LoadColumn(0), ColumnName("Label")] // exspeced perdicted out puts 
        public bool IsGood { get; set; }
       
        
        [LoadColumn(1)] // inputs 
        public string TxtUserFeedBack { get; set; }
    }
    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }
    public class Program
    {
       
        static List<FeedBackTrainingData> trainingData = new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData = new List<FeedBackTrainingData>();
        


        static void LoadTestData()
        {
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is good",
                IsGood = true
            });
            
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is better",
                IsGood = true
            });
            
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is average",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is nice",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is horrible",
                IsGood = false
            });
            
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is dumb",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is shit",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is crap",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is the worst",
                IsGood = false
            });
        }
        static void LoadTrainingData()
        {
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is really good",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is great",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is better",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is ok",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is average",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is nice",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is horrible",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is gay",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is dumb",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is shit",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is crap",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                TxtUserFeedBack = "this is the worst",
                IsGood = false
            });

        }
        public static void Main(string[] args)
        {

            // Steo 1 Load trainging data
            LoadTrainingData();
            // step 2 : create object of MLContext
            var mlContext = new MLContext( seed: 1);
            // step 3 convert data inn a Idataview
            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData> (trainingData);
            // step 4 : create pipeline define work flow in it 
            var pipeline = mlContext.Transforms
                            .Text.FeaturizeText("Features", "TxtUserFeedBack")
                                .Append(mlContext.BinaryClassification.Trainers.FastTree
                                    (numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));
            // step 5 training algoritm and we want the model out of it 
            var model = pipeline.Fit(dataView);
            
            // step 6: run test data to see accuracy
            LoadTestData();
            IDataView dataView1 = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(testData);

            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);


            // step 7  using the model


            //TestSinglePrediction(mlContext, model);
            //var predictionFuntion = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);
            string cont = "";

            while (cont == "")
            {
                Console.WriteLine("Entera Feedback string or enter n to stop");
                string feedbacksting = Console.ReadLine().ToString();
                var predictionFuntion = mlContext.Model.CreatePredictionEngine<FeedBackTrainingData, FeedBackPrediction>(model);

                var feedbackinput = new FeedBackTrainingData();
                feedbackinput.TxtUserFeedBack = feedbacksting;
                var feedbackpredicted = predictionFuntion.Predict(feedbackinput);
                Console.WriteLine("Predicted : - " + feedbackpredicted.IsGood);
                
            }
        }

       
    }
}
