

kernel  void convolve(global float *image, global Filter* filters, global float * featMap,int filterWidth,int inWidth,int featmapdim){
         
         const int xIn=get_global_id(0);//cols
         const int yIn=get_global_id(1);//rows
          
		  const int z=get_global_id(2);//filters
        // const int cur=get_group_id(0);
         
         float sum=0;
         for (int r=0;r<filterWidth;r++){
             for (int c=0;c<filterWidth;c++){
                 
                 sum+= filters[z].weights[c*filterWidth +r]*image[(xIn+c)*inWidth +(yIn+r)];
                 
                 
                 
            }
        }
        
		sum +=filters[z].bias;
        //featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=sum/(filterWidth*filterWidth);
         
         //featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=sigmoid(sum);     
		 switch(actflag){
			case 0: featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)] =sigmoid(sum);break;
			case 1: featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=mtanh(sum);break;
			case 2: featMap[(yIn+xIn*featmapdim +z*featmapdim*featmapdim)]=relu(sum);break;
		}
 }
         

 
  kernel void pooling( global float* prevfeatMap,global float* poolMap,global int* indexes,int Width,int pooldim){
 
 const int xIn=get_global_id(0);
 const int yIn=get_global_id(1);
 
  const int z=get_global_id(2);
// const int cur=get_group_id(0);
 

     float max=0;
	 int index=0;
         for (int r=0;r<2;r++){
             for (int c=0;c<2;c++){
                 
               
                 if(prevfeatMap[(xIn+c)*Width*z +(yIn+r)]>max){
                       max=prevfeatMap[(xIn+c)*Width*z +(yIn+r)];
					   index=c*2+r;
					   }
						
                 
                 }
             }
             
             poolMap[(yIn+xIn*pooldim +z*pooldim*pooldim)]=max;
			 indexes[(yIn+xIn*pooldim +z*pooldim*pooldim)]=index;
    
                                               
         
         
 
 }
 
 
 kernel void deltas(global Node * nodes,global Node * nextnodes,global float *deltas,global int *indexes,int dim,int nextnumNodes,int pooldim){
 
 const int xIn=get_global_id(0);
 const int yIn=get_global_id(1);
  const int z=get_global_id(2);

	
	int i=yIn+xIn*pooldim +z*pooldim*pooldim;
 
 
    float delta = 0;
    for (int j = 0; j !=nextnumNodes; j++)
        delta += nextnodes[j].delta * nextnodes[j].weights[i];
    //delta *= nodes[i].output*(1-nodes[i].output);
	switch(actflag){
		case 0: delta *= devsigmoid(nodes[i].output);break;
		case 1: delta *= devtanh(nodes[i].output);break;
		case 2: delta *= devrelu(nodes[i].output);break;
	}
	

	 for(int r=0;r<2;r++){
			for(int c=0;c<2;c++){
				if((c*2+r)==indexes[i])
					deltas[(2*yIn+r)+(2*xIn+c)*dim+z*dim*dim]=delta;
							
			}
	 
	 }



 }

 kernel void rotatemat( global float* source,global float* destin,int dim){
	
	const int xIn=get_global_id(0);
    const int yIn=get_global_id(1);

	destin[xIn*dim+yIn]=source[(dim-xIn)*dim+(dim-yIn)];
 
 }
 
 
   
kernel void backpropcnn( global float* featMap,global float* deltas,global Filter* filters,int featmapdim,int imagedim,int filterdim,float a,global float* Image){
 
 //todo: rotate  180 featmap//[i][j]->[dim -i][dim -j] if i,j>=0

         const int xIn=get_global_id(0);
         const int yIn=get_global_id(1);
		 const int z=get_global_id(2);
         
        
         
         float sum=0;
         for (int r=0;r<featmapdim;r++){
             for (int c=0;c<featmapdim;c++){
                 
                 sum+= deltas[(c+r*featmapdim +z*featmapdim*featmapdim)]*Image[(xIn+r)*imagedim +(yIn+c)];
                 
                 
                 
                 }
             }
          
        filters[z].weights[(xIn*filterdim +yIn)] -=a*sum;///(featmapdim*featmapdim) ;
          
       // filters[0].bias+=sum;///check this
        
         
         
                                               
         

 }


   
kernel void cnntoFcnn(global float* poolMap,global Node* nodes,int inputsize,int mapindex){


        const int xIn=get_global_id(0);
         const int yIn=get_global_id(1);
         
         const int z=get_global_id(2);
     
                 
        nodes[(yIn+xIn*inputsize +z*inputsize*inputsize)].output =poolMap[(yIn+xIn*inputsize +z*inputsize*inputsize)];
                   

 }


