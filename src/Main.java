import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
 
public class Main {
	class Parameters{
		double[] arr;
		double theta;
    	
    	public Parameters(double[] arr , double theta) {
    		this.arr = arr;
    		this.theta = theta;
    	}
    }//Parameters
	
    class BinaryNode {
        Parameters para;
        BinaryNode left;
        BinaryNode right;
 
        public BinaryNode(Parameters para) {
            this.para = para;
        }
    }//BinaryNode
    
    
    private static BinaryNode root;//根节点,这个属性很重要
    private static List<Parameters> list;
    
    public void buildCompleteTree(int depth , List<Parameters> list) {
    	int curDep = 1 , num = 1 , index = 0;
    	root = new BinaryNode(list.get(index++));
    	Queue<BinaryNode> queue = new LinkedList<>();
    	queue.add(root);
    	while(curDep < depth) {
    		int count = num * 2;
    		while(num < count) {
    			BinaryNode cur = queue.poll();
    			cur.left = new BinaryNode(list.get(index++));
    			queue.add(cur.left);
    			cur.right = new BinaryNode(list.get(index++));
    			queue.add(cur.right);
    			num += 2;
    		}
    		curDep++;
    	}
    	System.out.println("树构造完毕");
    }//buildCompleteTree
    
    public void printTreeLineByLine(BinaryNode root) {
        Queue<BinaryNode> queue = new LinkedList<BinaryNode>();//把待打印的结点放在队列中
        BinaryNode curNode;
        queue.offer(root);
        int curNum = 1;//当前行未打印的节点数
        int next = 0;//下一行待打印的节点数
        while (!queue.isEmpty()) {
            curNode = queue.poll();
            System.out.print("*");
            curNum--;
            if (curNode.left != null) {
                queue.offer(curNode.left);
                next++;
            }
            if (curNode.right != null) {
                queue.offer(curNode.right);
                next++;
            }
            if (curNum == 0) {
                curNum = next;
                next = 0;
                System.out.println();//换行
            }
        }//while
    }//printTreeLineByLine
    
    /*
     * row:向量维度（121） col:向量个数（310） k:每列非0元素个数
     */
    
    public Parameters[] produceVectorZ(int row , int col , int k){
    	Parameters[] ret = new Parameters[col];
    	Random rand = new Random();
    	
    	for(int j = 0 ; j < col ; j++) {
    		int num = k;
    		double[] x = new double[row];
    		while(num-- > 0) {
    			int ele = rand.nextInt(row - 1);
    			x[ele] = rand.nextInt(100) + 1; // 1 ~ 100
    			//System.out.print(" " + ele + "," + x[ele]);
    		}
    		int theta = rand.nextInt(80900) + 100;
    		//System.out.println(" " + theta);
    		ret[j] = new Parameters(x,theta);
    	}
    	System.out.println("总共的随机向量z以及theta产生完毕");
    	return ret;
    }
    
    /*
     * input:W * NumOfPatch , numOfRandZ:310随机z向量的个数 , candidates:候选参数数量
     */
    public void trainNode(double[][] input , int numOfRandZ , int candidates , Parameters[] sample , int depth) {
    	list = new ArrayList<Parameters>();
    	for(int i = 0 ; i < 1 << depth - 1 ; i++) {
	    	Parameters[] candidate = new Parameters[candidates];
	    	Random rand = new Random();
	    	double max = Double.MIN_NORMAL;
			int index = 0;
	    	for(int j = 0 ; j < candidates ; j++) {
	    		int randx = rand.nextInt(numOfRandZ);
	    		//System.out.println(randx);
	    		candidate[j] = sample[randx]; 
	    	}
	    	System.out.println("训练节点"+"i"+"候选z选择结束");

	    	for(int j = 0 ; j < candidates ; j++) {
	    		double tmp = computeI(input , candidate[j]);
	    		if(max < tmp) {
	    			index = j;
	    			max = tmp;
	    		}
	    	}
	    	list.add(candidate[index]);
	    	System.out.println("训练节点"+"i"+"结束");
    	}
    }
    
    public double computeI(double[][] input , Parameters para){
    	int Sl = 0 , Sr = 0 , S = input[0].length;
    	List<Integer> list = new ArrayList<>();
    	RealMatrix rm1 = new Array2DRowRealMatrix(input);
    	for(int j = 0 ; j < input[0].length; j++) {
    		RealMatrix z = new Array2DRowRealMatrix(para.arr);
    		if(z.transpose().multiply(rm1.getColumnMatrix(j)).getEntry(0, 0) - para.theta >= 0) {
    			Sl++;
    			list.add(j);
    		}
    		else
    			Sr++;
    	}
    	if(Sl == 0 || Sl == input[0].length)
    		return 0;
    	double[][] arrl = new double[input.length][Sl];
    	int indexl = 0 , indexr = 0;
    	double[][] arrr = new double[input.length][Sr];
    	for(int j = 0 ; j < input[0].length ; j++) {
    		if(list.contains(j)) {
    			for(int i = 0 ; i < input.length ; i++) {
    				arrl[i][indexl] = input[i][j];
    			}
    			indexl++;
    		}
    		else {
    			for(int i = 0 ; i < input.length ; i++) {
    				arrr[i][indexr] = input[i][j];
    			}
    			indexr++;
    		}
    	}
    	System.out.print("统计结束  ");
    	double[][] covl = new Covariance(new Array2DRowRealMatrix(arrl).transpose()).getCovarianceMatrix().getData();
    	double[][] covr = new Covariance(new Array2DRowRealMatrix(arrr).transpose()).getCovarianceMatrix().getData();
    	double[][] covs = new Covariance(new Array2DRowRealMatrix(input).transpose()).getCovarianceMatrix().getData();
    	System.out.print("协方差计算结束  ");
    	System.out.println(covs.length+","+covs[0].length);
    	double determinantOfCovl = determinant(covl);
    	System.out.print("行列式1计算结束  ");
    	double determinantOfCovr = determinant(covr);
    	System.out.print("行列式2计算结束  ");
    	double determinantOfS = determinant(covs);
    	System.out.println("行列式计算结束");
    	return Math.log(determinantOfS) - Sl/input[0].length*Math.log(determinantOfCovl) - Sr/input[0].length*Math.log(determinantOfCovr);
	}
   
    /*
     * 求行列式的值
     */
    static double determinant(double[][] a){  
        double result2 = 0;  
        if(a.length>2){  
         //每次选择第一行展开  
            for(int i=0;i<a[0].length;i++){  
                //系数符号  
                double f=Math.pow(-1,i);  
                //求余子式  
                double[][] yuzs=new double[a.length-1][a[0].length-1];  
                for (int j = 0; j < yuzs.length; j++) {  
                    for (int j2 = 0; j2 < yuzs[0].length; j2++) {  
                        //去掉第一行，第i列之后的行列式即为余子式  
                        if(j2<i){  
                            yuzs[j][j2]=a[j+1][j2];  
                        }else {  
                            yuzs[j][j2]=a[j+1][j2+1];  
                        }  
                          
                    }  
                }   
                //行列式的拉普拉斯展开式，递归计算  
                result2+=a[0][i]*determinant(yuzs)*f;  
            }  
        }  
        else{  
            //两行两列的行列式使用公式  
            if(a.length==2){  
                result2=a[0][0]*a[1][1]-a[0][1]*a[1][0];  
            }  
            //单行行列式的值即为本身  
            else{  
                result2=a[0][0];  
            }  
        }  
        return result2;  
    }
    
    public static List<String> readTxtFileIntoStringArrList(String filePath)
    {
        List<String> list = new ArrayList<String>();
        try
        {
            File file = new File(filePath);
            if (file.isFile() && file.exists())
            { // 判断文件是否存在
                InputStreamReader read = new InputStreamReader(
                        new FileInputStream(file));// 考虑到编码格式
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;

                while ((lineTxt = bufferedReader.readLine()) != null)
                {
                    list.add(lineTxt);
                }
                bufferedReader.close();
                read.close();
            }
            else
            {
                System.out.println("找不到指定的文件");
            }
        }
        catch (Exception e)
        {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
        }

        return list;
    }

    public static void main(String[] args) throws IOException {
    	Main main = new Main();
    	List<String> list1 = main.readTxtFileIntoStringArrList("mat/1.txt");
    	double[][] input = new double[121][100];
    	for(int i = 0 ; i < input.length ; i++) {
    		String[] arr = list1.get(i).split("\\s");
     		for(int j = 0 ; j < arr.length ; j++) {
    			input[i][j] = Double.parseDouble(arr[j]);
    		}
    	}
    	System.out.println("训练图像块读取完毕");
    	int depth = 1,W = 121,numOfRandZ = 310,candidates = 10,k = 4;
    	Parameters[] sample = main.produceVectorZ(W, numOfRandZ, k);
    	main.trainNode(input,numOfRandZ,candidates,sample,depth);
    	main.buildCompleteTree(depth,list);
    }
}