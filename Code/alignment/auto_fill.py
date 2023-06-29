from __init__ import *
import cv2
'''nhiệm vụ của file này còn phải tạo ra vùng bounding để bỏ những thông tin không nằm trong vùng chứa hóa đơn tổng
''' 

'''
padding một chút vùng đen xung quang hóa đơn để bao trọn được hết hóa đơn => sinh ra contour area'''

################### Global Variables##########################
padd_size = 30
threshold_area = 100000  # Set your desired area threshold
threshold_lenght = 1000
thresh_canny = [50,150]
##############################################################

def add_padding(img, padding_size= padd_size, padding_color=(0, 0, 0)):
    # Calculate the padding values
    top, bottom, left, right = padding_size, padding_size, padding_size, padding_size
    # Add padding to the image
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return pad_img

def dilate(img,padd = padd_size):
    h,w = img.shape[0:2]
    img = img[padd:h-padd,padd:w-padd]
    return img

def puring(img):
    ''''
    input: thresh image output of U2net

    -------------------

    output : mask of image after pured 2D value in {0,1} int
    '''
    img_cp = img.copy()
    img_cp = add_padding(img_cp) # add padding to get bill which cover the bouding of image
    gray = cv2.cvtColor(img_cp,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cny = cv2.Canny(thresh,*thresh_canny)
    cny = cv2.dilate(cny,np.ones((3,3)),iterations=1)
    ctn, _ = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(ctn, key=cv2.contourArea, reverse=True)
    filtered_contours = [contour for contour in sorted_contours if (cv2.arcLength(contour,False) > threshold_lenght and cv2.contourArea(contour) > threshold_area)]
    
    #### show area to choice threshold of area######3#
    # for ct in ctn:
    #     print(f'area : {cv2.contourArea(ct)}')
    #     print(f'lenght : {cv2.arcLength(ct,False)}')
    ##################################################

    mask = np.zeros(img_cp.shape[0:2]).astype(np.uint8) # h,w
    cv2.drawContours(mask,filtered_contours,0,(1),thickness=-1)
    return dilate(mask)
    

#mcocr_private_145120gmdua.jpg |normal
#mcocr_private_145120pxmgi.jpg |code bar down
#mcocr_private_145120btfhn.jpg |short
#mcocr_private_145120ddbdw.jpg |text out bill
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str
                        ,default=os.path.join(here,'Code/U2Net/output/mcocr_private_145120pxmgi.jpg'),
                        help="source image")
    otp = parser.parse_args()

    anh = cv2.imread(otp.source)
    name_image = os.path.splitext(os.path.basename(otp.source))[0]
    rgb = cv2.imread(os.path.join(here,'Vietnam_invoice_data/mcocr2021_raw/test/test_images',name_image+".jpg"))
    puring(anh,rgb)




