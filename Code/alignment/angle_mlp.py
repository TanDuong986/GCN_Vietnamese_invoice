'''
file này khảo sát để bỏ những hóa đơn không có một cạnh nào thẳng ,không tuân theo giới hạn để duỗi thằng
đồng thời bỏ các hóa đơn có độ tin cậy đánh nhãn không quá cao.
'''
from auto_fill import puring
from __init__ import *

'''
task: 
1: find all line approximate 4 line of bill (hough)
2: compare with threshold of angle'''
##### Global variable #########
Pi = 3.14159265359
variance_rho = 30
variance_theta = Pi/12
###############################

def norm_line(lines):
    '''
    Nếu góc trong khoảng từ 0- Pi/2 thì mặc định là góc theo trục Y
    
    Nếu góc trong khoảng từ Pi/2 - Pi (theo y) thì đổi góc về Pi - abs'''
    dist = lines[:,0]
    angle = lines[:,1]

    dist = np.array([dt if dt >0 else np.abs(dt) for dt in dist])
    angle = np.array([value if (value < Pi/2) else (Pi-value) for value in angle])
    return (dist,angle)

def lab(image):
    cl_img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(image, 1, np.pi/180, threshold=400) 
    '''
    rho là khoảng cách đến tâm (0,0), theta là góc tạo bởi đừng thẳng và trục Oy
    góc 0 dựng đứng, góc 90 nằm ngang, góc 180 độ chỉ xuống dưới'''
# Draw detected lines on the image
    
    if lines is not None:
        print(lines.shape)
        print(lines)
        lines = np.squeeze(lines)
        
        print()
        mask = norm_line(lines)
        x_coords = mask[0]
        y_coords = mask[1]
        # Plotting the points
        plt.scatter(x_coords,y_coords)

        # Adding labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Plotting Points')

        # Display the plot
        plt.show()
        
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
#mcocr_private_145120clklv.jpg |not enough
#mcocr_private_145120vogtr.jpg | not really line
if __name__ == "__main__":

    otp = gpr()
    path_img = os.path.join(here,'Code/U2Net/output',otp.source)
    print(path_img)
    anh = cv2.imread(path_img)
    name_image = os.path.splitext(os.path.basename(path_img))[0]
    rgb = cv2.imread(os.path.join(here,'Vietnam_invoice_data/mcocr2021_raw/test/test_images',name_image+".jpg"))
    mask,cny = puring(anh)
    lab(cny)
