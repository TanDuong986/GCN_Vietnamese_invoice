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
variance_rho = 60
variance_theta = 0.06
###############################

def norm_line(lines):
    '''
    Nếu góc trong khoảng từ 0- Pi/2 thì mặc định là góc theo trục Y
    
    Nếu góc trong khoảng từ Pi/2 - Pi (theo y) thì đổi góc về Pi - abs'''
    dist = lines[:,0]
    angle = lines[:,1]

    dist = np.array([dt if dt >0 else np.abs(dt) for dt in dist]).reshape(-1,1)
    # dist =np.array(dist).reshape(-1,1)
    angle = np.array([value if (value < Pi/2) else (Pi-value) for value in angle]).reshape(-1,1)
    # angle = np.array(angle).reshape(-1,1)
    rs = np.concatenate((dist,angle),axis = 1)
    return rs # shape (nx2)

def sort_line(lines): # use mask to compute the last line after fill, this will display all your feasible line
    mask = norm_line(lines)
    if mask.shape[0] == 1:
        return mask
    sm = mask[np.abs(mask[:, 0]).argsort()] # sorted_matrix by distance
    groups = []
    current_group = []

    for i in range(sm.shape[0] - 1):
        current_group.append(sm[i])
        diff = sm[i + 1, 0] - sm[i, 0]
        if diff >= variance_rho:
            groups.append(np.array(current_group))
            current_group = []

    current_group.append(sm[-1])
    groups.append(np.array(current_group))
    
    rs = np.array([group[0] for group in groups])
    # print(rs)
    return rs

def show_line_img(mask,cl_img,show_plt = False): ## draw line on image
    # print(mask.shape)
    if show_plt:
        x_coords = mask[:,0]
        y_coords = mask[:,1]
        # Plotting the points
        plt.scatter(x_coords,y_coords)
        # Adding labels and title
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Visualize of lines')
        # Display the plot
        plt.show()
    for rho, theta in mask: 
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 500*(-b))
        y1 = int(y0 + 500*(a))
        x2 = int(x0 - 500*(-b))
        y2 = int(y0 - 500*(a))
        cv2.line(cl_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # cv2.imwrite("line_detect.jpg",cl_img)
    show(cv2.resize(cl_img,(500,800)))

def detect_angle(image,show_plot = False):
    '''
    Input : Canny image

    Output: List of lines (rho,alpha)
    '''
    cl_img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(image, 1, np.pi/180, threshold=280) 
    # lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=10, minLineLength=100, maxLineGap=30)
    '''
    rho là khoảng cách đến tâm (0,0), theta là góc tạo bởi đừng thẳng và trục Oy
    góc 0 dựng đứng, góc 90 nằm ngang, góc 180 độ chỉ xuống dưới'''
    if lines is not None:
        lines = np.reshape(lines,(lines.shape[0],-1))
        mask = sort_line(lines)
        if show_plot:
            print(mask)
            print(lines)
            print(lines.shape[0],mask.shape[0])
            # print(mask)
            show_line_img(lines, cl_img,show_plot)
        return mask

def diagnose(lines):
    if lines is not None:
        accept = [0.0,Pi/2]
        agl = lines[:,1]
        rs = ((agl<=variance_theta) & (agl >=0)) | ((agl <= accept[1] + variance_theta) &(agl >= accept[1] - variance_theta))
        meann = np.mean(rs)
        if meann >= 0.5:
            return 0
        elif len(rs) == 1 :
            return -1
        else:
            return 1
    else : return -1
#mcocr_private_145120gmdua.jpg |normal |2 point
#mcocr_private_145120pxmgi.jpg |code bar down | 5 points
#mcocr_private_145120btfhn.jpg |short | 2 lines
#mcocr_private_145120ddbdw.jpg |text out bill | 3 point
#mcocr_private_145120clklv.jpg |not enough
#mcocr_private_145120vogtr.jpg | not really line

def embedd(img,mask):
    mask = np.expand_dims(mask, axis=-1)
    print(mask.shape)
    img = img*mask
    show(img)

if __name__ == "__main__":
    otp = gpr()
    path_img = os.path.join(here,'Code/U2Net/output',otp.source)
    anh = cv2.imread(path_img)
    name_image = os.path.splitext(os.path.basename(path_img))[0]
    rgb = cv2.imread(os.path.join(here,'Vietnam_invoice_data/mcocr2021_raw/mcocr_train_data/train_images',name_image+".jpg"))
    mask,cny = puring(anh) #create mask
    list_angle = detect_angle(cny,show_plot=True) #detect line
    # diagnose(list_angle)
    # embedd(rgb,mask)
