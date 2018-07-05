# SharpeningPhoto

## Result on 13 600 iteration
### training settings
- train data: 400 581 x 3 x 128 x 128 imgs
- val data: 17 808 x 3 x 128 x 128 imgs
- test data: 35 559 x 3 x 128 x 128 imgs
- batch_size: 224
- optimizer: Adam, lr=0.001
- loss: MSE

<table style="width:100%" align="center">
  <tr>
    <th>Blurred</th>
    <th>Sharp</th>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/8.jpg" height="250"/></td>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/8_sharp.jpg" height="250"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/1.JPG" height="250"/></td>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/1_sharp.JPG" height="250"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/2.JPG" height="250"/></td>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/2_sharp.JPG" height="250"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/3.JPG" height="250"/></td>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/3_sharp.JPG" height="250"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/4.jpg" height="250"/></td>
    <td align="center"><img src="https://github.com/0leynik/SharpeningPhoto/blob/master/release/input_img/4_sharp.jpg" height="250"/></td>
  </tr>
</table>
