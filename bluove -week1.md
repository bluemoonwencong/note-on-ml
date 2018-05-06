pathlist = "d:\\sync-mathematica\\data\\1.jpg";
ImageResize[Import[pathlist], {512, 512}]
img = ImageData@%;

![](https://github.com/bluove/note/blob/master/img/1525588327442.png)

![](https://github.com/bluove/note/blob/master/img/1525588377322.png)

activate[x_] := Max[0, x]
dactivate = UnitStep;
Plot[{activate[x], dactivate[x]}, {x, -2, 2}]

![](https://github.com/bluove/note/blob/master/img/1525588433626.png)

convlayer[input_, convkernel_, bias_, slide_] := 
 Block[{inputlenx, inputleny, inputlenz, convlenx, convleny, convlenz,
    outputlenx, outputleny, outputlenz, output, cachez, return},
  (*input=img;
  slide=1;*)
  {inputlenx, inputleny, inputlenz} = 
   Length /@ {input, First@input, First@First@input};
  {convlenx, convleny, convlenz} = 
   Length /@ {convkernel, First@convkernel, First@First@convkernel};
  {outputlenx, outputleny, 
    outputlenz} = {Floor@((inputlenx - convlenx + 1)/slide), 
    Floor@((inputleny - convleny + 1)/slide), inputlenz*convlenz};
  output = ConstantArray[0, {outputlenx, outputleny, outputlenz}];

  Do[

   cachez = 
    Total[input[[i + 0~Range~(convlenx - 1), 
        j + 0~Range~(convleny - 1), k]]*convkernel, 2] + 1*bias;
   output[[Floor@(i/slide) + 1, 
     Floor@(j/slide) + 1, (k - 1)*convlenz + 1~Range~convlenz]] = 
    activate /@ cachez;
   (*Print[{inputlenx,outputlenx,inputlenz,outputlenz,k,(k-1)*
   convlenz+1~Range~convlenz,cachez,i/slide,j/slide,Floor@(i/
   slide)}];*)

   , {k, 1, inputlenz}, {i, 1, inputlenx - convlenx + 1 - slide, 
    slide}, {j, 1, inputleny - convleny + 1 - slide, slide}];
  If[Mod[slide, 2] == 1, 
   return = output[[2~Range~outputlenx, 2~Range~outputleny]], 
   return = 
    output[[1~Range~(outputlenx - 1), 1~Range~(outputleny - 1)]]];
  return
  ]



pooling[input_, size_] := 
 Block[{inputlenx, inputleny, inputlenz, sizelenx, sizeleny, 
   outputlenx, outputleny, outputlenz, output, cachez},
  (*input=img;
  slide=1;*)
  {inputlenx, inputleny, inputlenz} = 
   Length /@ {input, First@input, First@First@input};
  {sizelenx, sizeleny} = size;
  {outputlenx, outputleny, outputlenz} = {inputlenx/sizelenx // Floor,
     inputleny/sizeleny // Floor, inputlenz};
  output = ConstantArray[0, {outputlenx, outputleny, outputlenz}];

  Do[

   cachez = 
    Mean@Mean@
      input[[i + 0~Range~(sizelenx - 1), j + 0~Range~(sizeleny - 1), 
       k]];
   output[[1 + i/sizelenx // Floor, 1 + j/sizeleny // Floor, k]] = 
    cachez;
   (*Print[{inputlenx,outputlenx,inputlenz,outputlenz,k,cachez}];*)

   , {k, 1, inputlenz}, {i, 1, inputlenx - sizelenx}, {j, 1, 
    inputleny - sizeleny}];
  output
  ]



convkernel = RandomReal[1, {3, 3, 4}];
bias1 = RandomReal[1, 4];
(*浮雕*)
convkernel[[;; , ;; , 1]] = {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
(*边缘强化*)
convkernel[[;; , ;; , 2]] = {{1, 1, 1}, {1, -7, 1}, {1, 1, 1}};
(*边缘检测*)
convkernel[[;; , ;; , 3]] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
convkernel[[;; , ;; , 4]];



output1 = convlayer[img, convkernel, bias1, 1];
output2 = pooling[output1, {3, 3}];

(Image@((1.5*#1[[;; , ;; , #2]])/Max@#1[[;; , ;; , #2]])) &[output1, 5]
Length /@ {#, First@#, First@First@#} & /@ {output1, output2}

![1525588510473](C:\Users\bluove\AppData\Local\Temp\1525588510473.png)

{{509, 509, 12}, {169, 169, 12}}



run := Block[{},
  convkernel = RandomReal[1, {3, 3, 4}];
  bias1 = RandomReal[1, 4];
  bias1 = RandomReal[1, 4];
  (*浮雕*)
  convkernel[[;; , ;; , 1]] = {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
  (*边缘强化*)
  convkernel[[;; , ;; , 2]] = {{1, 1, 1}, {1, -7, 1}, {1, 1, 1}};
  (*边缘检测*)
  convkernel[[;; , ;; , 3]] = {{-1, -1, -1}, {-1, 
     8, -1}, {-1, -1, -1}};
  convkernel[[;; , ;; , 4]];
  output1 = convlayer[img, convkernel, bias1, 1];
  output2 = pooling[output1, {3, 3}];

  convkernel3 = RandomReal[1, {3, 3, 4}];
  bias3 = RandomReal[1, 4];
  output3 = convlayer[output2, convkernel3, bias3, 1];
  output4 = pooling[output3, {3, 3}];

  convkernel5 = RandomReal[1, {3, 3, 20}];
  bias5 = RandomReal[1, 20];
  output5 = convlayer[output4, convkernel5, bias5, 1];
  output6 = pooling[output5, {3, 3}];

  convkernel7 = RandomReal[1, {3, 3, 4}];
  bias7 = RandomReal[1, 4];
  output7 = convlayer[output6, convkernel7, bias7, 1];
  output8 = pooling[output7, {2, 2}];

  dim8 = Length@Flatten@output8;
  dim9 = 1000;
  weight9 = RandomReal[0.0001, {dim9, dim8}];
  bias9 = RandomReal[1, dim9];
  output9 = activate /@ (weight9.Flatten@output8 + bias9);
  Length@output9;

  dim9 = Length@output9;
  dim10 = 50;
  weight10 = RandomReal[0.0001, {dim10, dim9}];
  bias10 = RandomReal[1, dim10];
  output10 = activate /@ (weight10.output9 + bias10);
  Length@output10;

  dim10 = Length@output10;
  dim11 = 2;
  weight11 = RandomReal[0.0001, {dim11, dim10}];
  bias11 = RandomReal[1, dim11];
  output11 = activate /@ (weight11.output10 + bias11);
  Length@output11;

  output11
  ]



targetfunc[selectnumber_] := If[selectnumber < 12500, {1, 0}, {0, 1}];
minifunc[selectnumber_] := Block[{},
   (*selectnumber=52;*)
   img = ImageData@
     Import["d:\\sync-cs\\bluoveGitHub\\kaggle\\train\\" <> 
       ToString[selectnumber] <> ".jpg"];
   run];

select = RandomSample[Range@25000, 3];
mini = minifunc /@ select
target = minifunc /@ select



cost = 1/2*Total@Total@((mini - target)^2)



输入图片是512 * 512 * 3，第一层卷积核用3 * 3 *4，输出output1维度509 * 509 *12，第二层pooling输出output2维度169 * 169 *12.

img（input，dim＝512 * 512 * 3） 

-> conv（output1，dim＝509 * 509 * 12） 

->pooling（output2，dim＝169 * 169 * 12） 

-> conv（output3，dim＝166 * 166 * 48） 

->pooling（output4，dim＝55 * 55 * 48） 

-> conv（output5，dim＝52 * 52 * 960） 

->pooling（output6，dim＝17 * 17 * 960） 

-> conv（output7，dim＝14 *14 * 3840） 

->pooling（output8，dim＝7 * 7 * 3840） 

-> full（output9，dim＝1000）

-> full（output10，dim＝50）

->full（output11，dim＝2）

