import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

public class Test {
	public static void main(String[] args) {
		int depth = 5,W = 121,numOfRandZ = 1000,candidates = 50,k = 4;
		List<String> listN = Main.readTxtFileIntoStringArrList("/home/zhangqi/UltraStereo/UltraStereo/nodePara/sbN" + depth + "_" + numOfRandZ + "_" + candidates + ".txt");
		List<String> listTheta = Main.readTxtFileIntoStringArrList("/home/zhangqi/UltraStereo/UltraStereo/nodePara/sbTheta" + depth + "_" + numOfRandZ + "_" + candidates + ".txt");
		List<String> listInput = Main.readTxtFileIntoStringArrList("/home/zhangqi/UltraStereo/UltraStereo/mat/_WL1.txt");
		List<String> WRALL = Main.readTxtFileIntoStringArrList("/home/zhangqi/UltraStereo/UltraStereo/mat/_WR1.txt");
		List<Parameters> list = new ArrayList<>();
		String[] arrTheta = listTheta.get(0).split("\\s");
		for(int i = 0 ; i < arrTheta.length ; i++) {
			String[] arrN = listN.get(i).split("\\s");
			double[] arr = new double[121];
			for(int j = 0 ; j < W ; j++)
				arr[j] = Double.parseDouble(arrN[j]);
			list.add(new Parameters((arr),Double.parseDouble(arrTheta[i])));
		}
		double[] WL1 = new double[W];
		for(int i = 0 ; i < listInput.size() ; i++) {
			WL1[i] = Double.parseDouble(listInput.get(i));
		}
		double[][] arrWRALL = new double[W][WRALL.get(0).split("\\s").length];
		for(int i = 0 ; i < arrWRALL.length ; i++) {
    		String[] arr = WRALL.get(i).split("\\s");
    		for(int j = 0 ; j < arr.length ; j++) {
    			arrWRALL[i][j] = Double.parseDouble(arr[j]);
    		}
		}
		int[] y_WL1 = Test2(WL1 , depth , list);
		RealMatrix rmWRALL = new Array2DRowRealMatrix(arrWRALL);
		rmWRALL = rmWRALL.transpose();
		int index = 0;
		int distance = 100;
		List<String> record = new ArrayList<String>();
		int[] recordD = new int[arrWRALL[0].length];
		for(int i = 0 ; i < arrWRALL[0].length ; i++) {
			double[] tmp = rmWRALL.getRow(i);
			int[] tmpY = Test2(tmp , depth , list);
			System.out.println(Arrays.toString(tmpY));
			record.add(Arrays.toString(tmpY));
			int tmpD = Main.hammingDistance(y_WL1, tmpY);
			recordD[i] = tmpD;
			if(tmpD < distance) {
				distance = tmpD;
				index = i;
			}
			
		}
		System.out.println("match patch : " + index + " ,  hamming : " + distance);
		System.out.println("total index :");
		for(int i = 0 ; i < recordD.length ; i++) {
			if(recordD[i] == distance) {
				System.out.print(i + " ,");
			}
		}
		System.out.println();
		System.out.println("WL's Encoding : ");
		System.out.println(Arrays.toString(y_WL1));
	}
	
	/*
	 * x:需要编码的图像块，depth:二叉树的深度，list:二叉树节点的参数列表
	 * 输出y:31位(0,1)编码
	 */
	public static int[] Test1(double[] x , int depth , List<Parameters> list) {
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
	
	/*
	 * x:需要编码的图像块，depth:二叉树的深度，list:二叉树节点的参数列表
	 * 层次遍历而不是类似与决策树那种
	 */
	
	public static int[] Test2(double[] x , int depth , List<Parameters> list) {
		int[] y = new int[(1 << depth) - 1];
		RealMatrix input = new Array2DRowRealMatrix(x);
		for(int i = 0 ; i < y.length ; i++) {
			Parameters node = list.get(i);
			double t = 0;
			for(int j = 0 ; j < x.length ; j++)
	    		if(node.arr[j] != 0)
	    			t += x[j];
			RealMatrix rm1 = new Array2DRowRealMatrix(node.arr);
    		double theta = node.theta;
    		//System.out.println(input.transpose().multiply(rm1).getEntry(0, 0) + "*****" + theta * t);
    		if(input.transpose().multiply(rm1).getEntry(0, 0) - theta * t >= 0)
    			y[i] = 1;
    		else
    			y[i] = 0;
    		
		}
		return y;
	}
}
