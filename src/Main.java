import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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
	
    private static BinaryNode root;//根节点,这个属性很重要
    private static List<Parameters> list;
    private static List<Integer> listNum;
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
     * row:向量维度（121） col:向量个数（1000） k:每列非0元素个数
     */
    
    public Parameters[] produceVectorZ(int row , int col , int k){
    	Parameters[] ret = new Parameters[col];
    	Random rand = new Random();
    	for(int j = 0 ; j < col ; j++) {
    		double sum = 0;
    		int num = k;
    		double[] x = new double[row];
    		while(num-- > 0) {
    			int ele = rand.nextInt(row - 1);
    			x[ele] = rand.nextDouble();
    			sum+=x[ele];
    		}
    		double theta = sum / (4 + 1 - rand.nextDouble() * 2);
    		//System.out.println(theta + " , " + sum);
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
    	listNum = new ArrayList<Integer>();
    	int[] numRecord = new int[candidates];
    	for(int i = 0 ; i < (1 << depth) - 1 ; i++) {
	    	Parameters[] candidate = new Parameters[candidates];
	    	Random rand = new Random();
	    	double max = Double.MIN_NORMAL;
			int index = 0;
	    	for(int j = 0 ; j < candidates ; j++) {
	    		int randx = rand.nextInt(numOfRandZ);
	    		//System.out.println(randx);
	    		candidate[j] = sample[randx]; 
	    		numRecord[j] = randx;
	    	}

	    	for(int j = 0 ; j < candidates ; j++) {
	    		double tmp = computeI(input , candidate[j]);
	    		if(max < tmp) {
	    			index = j;
	    			max = tmp;
	    		}
	    	}
	    	listNum.add(numRecord[index]);
	    	list.add(candidate[index]);
	    	//System.out.println(listNum.toString());
	    	//System.out.println("训练节点"+i+"结束");
    	}
    }
    
    public double computeI(double[][] input , Parameters para){
    	int Sl = 0 , Sr = 0 , S = input[0].length;
    	List<Integer> list = new ArrayList<>();
    	RealMatrix rm1 = new Array2DRowRealMatrix(input);
    	double[] arr1 = new double[input.length];
    	for(int i = 0 ; i < arr1.length ; i++)
    		if(para.arr[i] != 0)
    			arr1[i] = 1;
    	RealMatrix one = new Array2DRowRealMatrix(arr1);
    	for(int j = 0 ; j < input[0].length; j++) {
    		RealMatrix z = new Array2DRowRealMatrix(para.arr);
    		double tmp = z.transpose().multiply(rm1.getColumnMatrix(j)).getEntry(0, 0);
    		//System.out.println(tmp - para.theta);
    		double x = one.transpose().multiply(rm1.getColumnMatrix(j)).getEntry(0, 0) / 4;
    		if(tmp - para.theta * x  * 4 >= 0) {
    			Sl++;
    			list.add(j);
    		}
    		else
    			Sr++;
    	}
    	//System.out.println(Sl + " , " + Sr);
    	if(Sl <=1 || Sr <= 1)
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
    	double[][] covl = new Covariance(new Array2DRowRealMatrix(arrl).transpose()).getCovarianceMatrix().getData();
    	double[][] covr = new Covariance(new Array2DRowRealMatrix(arrr).transpose()).getCovarianceMatrix().getData();
    	//double determinantOfCovl = determinant(covl);
    	double determinantOfCovl = GetLineTran(covl,121);
    	//double determinantOfCovr = determinant(covr);
    	double determinantOfCovr = GetLineTran(covr,121);
    	//double determinantOfS = determinant(covs);
    	double determinantOfS = GetLineTran(covr,121);
    	return Math.log(determinantOfS) - Sl/input[0].length*Math.log(determinantOfCovl) - Sr/input[0].length*Math.log(determinantOfCovr);
	}
   
    /*
     * 求行列式的值
     */
    
    public static double GetLineTran(double[][] p, int n) {
		if (n == 1) return p[0][0];
		
		double exChange = 1.0; // 记录行列式中交换的次数
		boolean isZero = false; // 标记行列式某一行的最右边一个元素是否为零
 
		for (int i = 0; i < n; i ++) {// i 表示行号
			if (p[i][n - 1] != 0) { // 若第 i 行最右边的元素不为零
				isZero = true;
				
				if (i != (n - 1)) { // 若第 i 行不是行列式的最后一行
					for (int j = 0; j < n; j ++) { // 以此交换第 i 行与第 n-1 行各元素
						double temp = p[i][j];
						p[i][j] = p[n - 1][j];
						p[n - 1][j] = temp;
						
						exChange *= -1.0;
					}
				}
				
				break;
			}
		}
		
		if (!isZero) return 0; // 行列式最右边一列元素都为零，则行列式为零。
		
		
		for (int i = 0; i < (n - 1); i ++) {
		// 用第 n-1 行的各元素，将第 i 行最右边元素 p[i][n-1] 变换为 0，
		// 注意：i 从 0 到 n-2，第 n-1 行的最右边元素不用变换
			if (p[i][n - 1] != 0) {
				// 计算第  n-1 行将第 i 行最右边元素 p[i][n-1] 变换为 0的比例
				double proportion = p[i][n - 1] / p[n - 1][n - 1];
				
				for (int j = 0; j < n; j ++) {
					p[i][j] += p[n - 1][j] * (- proportion);
				}
			}
		}
		
		return exChange * p[n - 1][n - 1] * GetLineTran(p, (n - 1));
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
    
    public static int hammingDistance(int[] y1 , int[] y2) {
    	int num = 0;
    	for(int i = 0 ; i < y1.length ; i++) {
    		if((y1[i] ^ y2[i]) == 1) {
    			num++;
    		}
    	}
    	
    	return num;
    }
    
    /**
	    * 把字符串写入文本中
	    * @param fileName 生成的文件绝对路径
	    * @param content 文件要保存的内容
	    * @param enc  文件编码
	    * @return
    */
    public static boolean writeStringToFile(String fileName,String content,String enc) {
        File file = new File(fileName);
        try {
            if(file.isFile()){
                file.deleteOnExit();
                file = new File(file.getAbsolutePath());
            }
            OutputStreamWriter os = null;
            if(enc==null||enc.length()==0){
                os = new OutputStreamWriter(new FileOutputStream(file));
            }else{
                os = new OutputStreamWriter(new FileOutputStream(file),enc);
            }
            os.write(content);
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }
    public static void main(String[] args) throws IOException {
    	Main main = new Main();
    	List<String> list1 = Main.readTxtFileIntoStringArrList("mat/1.txt");
    	//List<String> list2 = main.readTxtFileIntoStringArrList("mat/2.txt");
    	//List<String> list3 = main.readTxtFileIntoStringArrList("mat/3.txt");
    	//List<String> list4 = main.readTxtFileIntoStringArrList("mat/4.txt");
    	//List<String> list5 = main.readTxtFileIntoStringArrList("mat/5.txt");
    	//List<String> list6 = main.readTxtFileIntoStringArrList("mat/6.txt");
    	double[][] input = new double[121][list1.get(0).split("\\s").length];
    	//double[][] input2 = new double[121][100];
    	//double[][] input3 = new double[121][100];
    	//double[][] input4 = new double[121][100];
    	//double[][] input5 = new double[121][100];
    	//double[][] input6 = new double[121][100];
    	for(int i = 0 ; i < input.length ; i++) {
    		String[] arr = list1.get(i).split("\\s");
    		//String[] arr2 = list2.get(i).split("\\s");
    		//String[] arr3 = list3.get(i).split("\\s");
    		//String[] arr4 = list4.get(i).split("\\s");
    		//String[] arr5 = list5.get(i).split("\\s");
    		//String[] arr6 = list6.get(i).split("\\s");
     		for(int j = 0 ; j < arr.length ; j++) {
    			input[i][j] = Double.parseDouble(arr[j]);
    			//input2[i][j] = Double.parseDouble(arr2[j]);
    			//input3[i][j] = Double.parseDouble(arr3[j]);
    			//input4[i][j] = Double.parseDouble(arr4[j]);
    			//input5[i][j] = Double.parseDouble(arr5[j]);
    			//input6[i][j] = Double.parseDouble(arr6[j]);
    		}
    	}
    	System.out.println("训练图像块读取完毕");
    	int depth = 5,W = 121,numOfRandZ = 1000,candidates = 50,k = 4;
    	Parameters[] sample = main.produceVectorZ(W, numOfRandZ, k);
    	main.trainNode(input,numOfRandZ,candidates,sample,depth);
    	main.buildCompleteTree(depth,list);
    	StringBuilder sbN = new StringBuilder();
    	StringBuilder sbTheta = new StringBuilder();
    	for(int j = 0 ; j < list.size() ; j++) {
    		double[] tmp = list.get(j).arr;
    		sbTheta.append(list.get(j).theta);
    		if(j == list.size() - 1)
    			sbTheta.append("\n");
    		else
    			sbTheta.append(" ");
    		for(int i = 0 ; i < W - 1 ; i++) {
    			sbN.append(tmp[i] + " ");
    		}
    		sbN.append(tmp[W - 1] + "\n");
    	}
    	
    	if(Main.writeStringToFile("/home/zhangqi/UltraStereo/UltraStereo/nodePara/sbTheta" + depth + "_" + numOfRandZ + "_" + candidates + ".txt", sbTheta.toString(), null)) {
    		System.out.println("存取Theta成功");
    	}
    	else {
    		System.out.println("存取Theta失败");
    	}
    	if(Main.writeStringToFile("/home/zhangqi/UltraStereo/UltraStereo/nodePara/sbN" + depth + "_" + numOfRandZ + "_" + candidates + ".txt", sbN.toString(), null)) {
    		System.out.println("存取N成功");
    	}
    	else {
    		System.out.println("存取N失败");
    	}
    	/*
    	System.out.println("*********************************Test*********************************");
    	RealMatrix input_2 = new Array2DRowRealMatrix(input2);
    	int[] y1 = main.Test(input_2.getColumn(0), depth);
    	System.out.println(Arrays.toString(y1));
    	int[] y2 = main.Test(input_2.getColumn(1), depth);
    	System.out.println(Arrays.toString(y2));
    	System.out.println(main.hammingDistance(y1, y2));
    	*/
    }
}