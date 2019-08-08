import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class Test {
	public static void main(String[] args) {
		List<String> listN = Main.readTxtFileIntoStringArrList("/home/zhangqi/eclipse-workspace/UltraStereo/nodePara/sbN5.txt");
		List<String> listTheta = Main.readTxtFileIntoStringArrList("/home/zhangqi/eclipse-workspace/UltraStereo/nodePara/sbTheta5.txt");
		List<String> listInput = Main.readTxtFileIntoStringArrList(" ");
		List<Parameters> list = new ArrayList<>();
		String[] arrTheta = listTheta.get(0).split("\\s");
		for(int i = 0 ; i < arrTheta.length ; i++) {
			String[] arrN = listN.get(i).split("\\s");
			double[] arr = new double[121];
			for(int j = 0 ; j < 121 ; j++)
				arr[j] = Double.parseDouble(arrN[j]);
			list.add(new Parameters(arr,Double.parseDouble(arrTheta[i])));
		}
		
	}
	
	/*
	 * x:需要编码的图像块，depth:二叉树的深度，list:二叉树节点的参数列表
	 * 输出y:31位(0,1)编码
	 */
	public static int[] Test(double[] x , int depth , List<Parameters> list) {
    	int[] y = new int[(1 << depth) - 1];
    	int i = 1;
    	RealMatrix input = new Array2DRowRealMatrix(x);
    	while(depth-- >= 1) {
    		y[i - 1] = 1;
    		Parameters node = list.get(i - 1);
    		RealMatrix z = new Array2DRowRealMatrix(node.arr);
    		double theta = node.theta;
    		//System.out.println(input.getColumnMatrix(0).transpose().multiply(z).getEntry(0,0) - theta);
    		if(input.transpose().multiply(z).getEntry(0,0) - theta >= 0) {
    			i *= 2;
    		}
    		else {
			if(i == 1) {
			    y[i - 1] = 0;			
			}
    			i = 2 * i + 1;
    		}
    	}
    	return y;
    }
}
