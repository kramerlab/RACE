import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;
/**
 * it calculates the average of all the measures for any number of runs but only one classifier
 * the files should be in the format of: pathOfMethod/.../measure/runNumber.txt
 * 
 * edit: on 24.4.2017 I added the averaging for offline experiment as well. 
 * Now the first line is the results for online RACE-AE & second line for offline variant
 *
 */

public class AverageOfMeasures {
	static ArrayList<File> allFiles;
	static ArrayList<ArrayList<File>> FilesforRuns;
	static ArrayList<String> measures = new ArrayList<String>();
	static int runs = 0;
//	static ArrayList<String> alg = new ArrayList<String>();
	
	public static void main(String[] args) throws IOException {
		//make the averages of all measures (the folder is outputPath)
//		System.out.println(args[0]);
//		if(args[0].endsWith("/"))
//			args[0] = args[0].substring(0, args[0].length()-1);
		String path = "/media/havij/dataDrive/RACE-experimental results/results-lenovo-combinedWithDropbox/rcv1v2subset1_500_NaiveBayesUpdateable/7/repeat-RACE";
//		String path = "/home/havij/Dropbox/Backup Codes/results/rcv1v2subset1_500_NaiveBayesUpdateable/7/repeat-RACE";
//		String path = "/home/havij/Dropbox/Backup Codes/results/batchIters/mediamill_500_NaiveBayesUpdateable/7/repeat-RACE";
//		String path = "/home/havij/Dropbox/Backup Codes/results/nus-wide-bow_500_NaiveBayesUpdateable/7/repeat-RACE";
		AverageOfMeasures am = new AverageOfMeasures();
		am.calculateAvgOfOneMethod(path,true);
	}
	
	public void calculateAvgOfOneMethod(String args, boolean offline) throws IOException {	
		String datasetPath = args;
//		boolean offlineFlag = false;
		
		allFiles = new ArrayList<File>();
		FilesforRuns = new ArrayList<ArrayList<File>>();
		returnFilePath(datasetPath);
		PrintWriter pw;
		double[] TotAvgValues = new double[measures.size()];
		double[][] avgPerRun = new double[measures.size()][runs];
//		double[] stdValues = new double[measures.size()];
		double[] offValues = new double[measures.size()];
		int[] numValues = new int[measures.size()];
		
		
		
		
		for(int i=0; i<allFiles.size(); i++){
			String path = allFiles.get(i).getPath();
			if(!path.contains("threshold") && !path.contains("BetaSeq")){
				String r = path.substring(path.lastIndexOf("/")+1, path.lastIndexOf(".")); //run
				String f = path.substring(0,path.lastIndexOf("/"));	//folder
				String meas = f.substring(f.lastIndexOf("/")+1, f.length());	//measure
				int ind = measures.indexOf(meas);
				
				BufferedReader bf = new BufferedReader(new FileReader(allFiles.get(i)));
				String line;
				if(!offline){
					while((line = bf.readLine())!= null){
						numValues[ind]++;
						TotAvgValues[ind] += new Double(line);
						avgPerRun[ind][new Integer(r)-1] += new Double(line);
					}
				}else{ 
					while((line = bf.readLine())!= null){
						StringTokenizer stg = new StringTokenizer(line, ",");
						if(stg.countTokens() == 3) { // offlineCompare
							String off = stg.nextToken().trim();
							if(off.length() > 0)
								offValues[ind] = new Double(off); //ignore offline value							
						}
						String val = stg.nextToken().trim();
						if(val.length() > 0){
							numValues[ind]++;
							TotAvgValues[ind] += new Double(val);
							avgPerRun[ind][new Integer(r)-1] += new Double(val);
						}
					}
				}
				
			}
		}
		
		
		//write averages in file
		File f = new File(datasetPath+"-Avg/");
		f.mkdirs();
		pw = new PrintWriter(new File(datasetPath+"-Avg/average.txt"));
		for(int k=0; k<measures.size(); k++){
			if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq"))
				pw.print(measures.get(k).replaceAll(" ", "-")+" & ");
		}
		pw.println();
		
		for(int k=0; k<TotAvgValues.length; k++){
			if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq")){
				TotAvgValues[k] = TotAvgValues[k]/numValues[k];
//				System.out.println(numValues[k]+"**\t");
				for(int rr=0; rr<runs; rr++) {
					avgPerRun[k][rr] = avgPerRun[k][rr]*runs/numValues[k];
//					System.out.print(avgPerRun[k][rr]+",");
				}
				System.out.println(TotAvgValues[k]+" "+getMean(avgPerRun[k])+"+"+getStdDev(avgPerRun[k]));
				pw.print(String.format( "%.2f", getMean(avgPerRun[k])) + "+"+String.format( "%.2f", getStdDev(avgPerRun[k]))+ " & ");
			}
		}
		pw.println();
		if(offline){ //print the offline RACE results
			for(int k=0; k<offValues.length; k++){
				if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq")){
					pw.print(String.format( "%.2f", offValues[k]) + " & ");
				}
			}
		}
		pw.close();
	
	}
	
	
	public double getMean(double[] AVGperRun) {
        double sum = 0.0;
        for(double a : AVGperRun)
            sum += a;
        return sum/AVGperRun.length;
    }
	
	public double getVariance(double[] AVGperRun) {
        double mean = getMean(AVGperRun);
        double temp = 0;
        for(double a :AVGperRun)
            temp += (a-mean)*(a-mean);
        return temp/(AVGperRun.length-1);
    }

    double getStdDev(double[] AVGperRun) {
        return Math.sqrt(getVariance(AVGperRun));
    }
	
	
	public void returnFilePath(String originalPath){
		File[] dir = new File(originalPath).listFiles();
		for(int i=0; i<dir.length; i++){
			if(dir[i].isDirectory()){
				returnFilePath(dir[i].getAbsolutePath());
			}else{
				String p = dir[i].getAbsolutePath();	//path
				if(!p.contains("thresholds") && !p.contains("misclass") && !p.contains("BetaSeq")){
					allFiles.add(dir[i]);
					String r = p.substring(p.lastIndexOf("/")+1, p.lastIndexOf(".")); //run 
					String f = p.substring(0,p.lastIndexOf("/"));	//folder
					String m = f.substring(f.lastIndexOf("/")+1, f.length());	//measure
					if(!measures.contains(m)) {
						measures.add(m);
					}
					if(new Integer(r) > runs) {
						runs = new Integer(r);
						FilesforRuns.add(new ArrayList<File>());
					}
					FilesforRuns.get(new Integer(r)-1).add(dir[i]);
				}
			}
		}
	}
}
