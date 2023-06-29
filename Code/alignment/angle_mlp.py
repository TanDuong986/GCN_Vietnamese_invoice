'''
file này khảo sát để bỏ những hóa đơn không có một cạnh nào thẳng ,không tuân theo giới hạn để duỗi thằng
đồng thời bỏ các hóa đơn có độ tin cậy đánh nhãn không quá cao.
'''
from auto_fill import puring
from __init__ import *
from Buddha.endless_source_of_light import omm

'''
task: 
1: find all line approximate 4 line of bill (hough)
2: compare with threshold of angle'''

##### Global variable #########
Pi = 1.57079632679
variance_rho = 30
variance_theta = Pi/6
###############################
def lab(image):
    cl_img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(image, 1, np.pi/180, threshold=200)
    print(len(lines))
# Draw detected lines on the image
    if lines is not None:
        lines = np.squeeze(lines)
        print(lines)
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 600*(-b))
            y1 = int(y0 + 600*(a))
            x2 = int(x0 - 600*(-b))
            y2 = int(y0 - 600*(a))

            cv2.line(cl_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show(cv2.resize(cl_img,(500,800)))


#mcocr_private_145120gmdua.jpg |normal
#mcocr_private_145120pxmgi.jpg |code bar down
#mcocr_private_145120btfhn.jpg |short
#mcocr_private_145120ddbdw.jpg |text out bill
if __name__ == "__main__":

    otp = gpr()
    path_img = os.path.join(here,'Code/U2Net/output',otp.source)
    print(path_img)
    anh = cv2.imread(path_img)
    name_image = os.path.splitext(os.path.basename(path_img))[0]
    rgb = cv2.imread(os.path.join(here,'Vietnam_invoice_data/mcocr2021_raw/test/test_images',name_image+".jpg"))
    mask,cny = puring(anh)
    lab(cny)
