package prepration;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

public class writeDatasetLabelsInFile {

	public static void main(String[] args) throws InvalidDataFormatException, FileNotFoundException {
		MultiLabelInstances dataset = new MultiLabelInstances(args[0]+".arff", args[0]+".xml");
		int[] labelInx = dataset.getLabelIndices();
		Instances inst = dataset.getDataSet();
		
		PrintWriter pw = new PrintWriter(args[0]+"-labels.txt");
		
		for(int i=0; i<inst.numInstances();i++){
			for(int fi=0; fi<labelInx.length; fi++){
				if(fi == labelInx.length-1)
					pw.print(inst.instance(i).value(labelInx[fi])+"\n");
				else
					pw.print(inst.instance(i).value(labelInx[fi])+",");
			}
		}
		
		pw.close();
	}

}
