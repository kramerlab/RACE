import java.util.ArrayList;
import java.util.Random;

public class gramschmidt {
	
	Random rand = new Random();
	public static double[][] basisVector;
	static int numVectors, dimensions;
	
	//rows = hidden , cols = input + 1 (for bias!)
	public gramschmidt(int rows, int columns){
		numVectors = rows;
		dimensions = columns;
		
//		RealMatrix h0 = MatrixUtils.createRealMatrix(rows, columns); //elements are set to 0
		basisVector = new double[rows][columns];
		
		for(int i=0; i<rows; i++){
			for(int j=0; j<columns; j++){
				double rnd = rand.nextDouble();
				basisVector[i][j] = 2*rnd-1;
			}
		}
	}
	
	
	public static double[][] makeOrthogonals(){
		double[][] orthoVectors = new double[numVectors][dimensions];
		
		ArrayList<double[]> vectorList = new ArrayList<double[]>();
		for(int i=0; i<numVectors; i++){
			vectorList.add(basisVector[i]);
		}
		
		for(int i=0; i<numVectors; i++){
			double[] currentVector = vectorList.get(i);
			double mag = magnitude(currentVector);
			for(int j=0; j<currentVector.length; j++)
				currentVector[j] /= mag;
			vectorList.set(i, currentVector);
			
			for(int j=i+1; j<numVectors; j++){
				double[] nextVector = vectorList.get(j);
				double[] projectNC = projected(nextVector, currentVector);
				for(int k=0; k<nextVector.length; k++)
					nextVector[k] = nextVector[k]-projectNC[k];
				vectorList.set(j, nextVector);
			}
		}
		
		for(int i=0; i<vectorList.size(); i++)
			orthoVectors[i] = vectorList.get(i);
		
		return orthoVectors;
	}
	
	//project vector v on u (proj_u(v))
	public static double[] projected(double[] v, double[] u){
		double[] proj = new double[v.length];
		
		//if u=0 then proj=0
		double dotUV = dot(u,v);
		double dotUU = dot(u,u);
		for(int i=0; i<u.length;i++)
			proj[i] = dotUV * u[i] / dotUU;
		
		return proj;
	}
	
	 // return the inner product of Vector a and b
    public static double dot(double[] v1, double[] v2) {
        if (v1.length != v2.length) 
        	throw new RuntimeException("Dimensions don't agree");
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++)
            sum = sum + (v1[i] * v2[i]);
        return sum;
    }

    // return the Euclidean norm of Vector
    public static double magnitude(double[] v) {
        return Math.sqrt(dot(v,v));
    }
    
    
    //test
    public static void main(String[] args) {
		double[][] test = new double[][]{{3.0,1.0} , {2.0,2.0}};
		basisVector = test;
		numVectors = dimensions = 2;

		for(int i=0; i<basisVector.length; i++){
			for(int j=0; j<basisVector[i].length;j++)
				System.out.print(basisVector[i][j]+" ");
			System.out.println();
		}
		
		double[][] output = makeOrthogonals();
		
		System.out.println("hello!");
		for(int i=0; i<output.length; i++){
			for(int j=0; j<output[i].length;j++)
				System.out.print(output[i][j]+" ");
			System.out.println();
		}
	}
	
}
