namespace DigitRecognizer
{
    class Program
    {
		public static void Main(string[] args)
		{
			string algorithm = args[0];

			// all three algorithms process data very differently
			// they exist in different classes
			switch (algorithm)
			{
				case "knn":
					Knn.Run();
					break;
				case "linear":
					OnlineLms.Run();
					break;
				case "nb":
					NaiveBayes.Run();
					break;
			}
		}
	}
}
