#include <cv.h>
#include <highgui.h>

using namespace cv;

Vec3b Red( 0, 0, 255 );

struct Location{
    int X, Y, size;

    Location( unsigned int a, unsigned int b, unsigned int c ) : X( a ), Y( b ), size( c ) {}

    const bool operator==(const Location& other) const {
        return X == other.X && Y == other.Y;
    }
};

const int VectorSize( const std::vector<int>& data ){
    int retVal = 0;
    for( int i = 0; i < data.size(); i++ ) retVal += data[i];
    return retVal;
}

const bool isFinder( const int size, const std::vector<int>& data ){
    const int partSize = size / 7;
    const int tolerance = partSize / 2;

    return std::abs( partSize - data[0] ) < tolerance && std::abs( partSize - data[1] ) < tolerance &&
        std::abs(  3 * partSize - data[2] ) <  2 * tolerance && //3 * 3 *
        std::abs( partSize - data[3] ) < tolerance && std::abs( partSize - data[4] ) < tolerance;
}

const std::vector<Location> FindMarksHorizontal( Mat image, const int maxfeature ){
    std::vector<Location> retVal;

    for( unsigned int i = 0; i < image.rows; i++ ){
        for( unsigned int j = 0; j < image.cols; j++ ){
            if( image.at<unsigned char>(i, j) < 128 ) {
                int current = 0;
                std::vector<int> data( 5, 0 );

                for( unsigned int k = j - 1; k < image.cols; k++ ){
                    unsigned char value = image.at<unsigned char>(i, k);

                    if( current % 2 == 0 ) {
                        if( value < 128 ) {
                            data[current]++;
                            if(data[current] > maxfeature){
                                j = max(j, k);
                                break;
                            }
                        }else{
                            current++;
                        }
                    }else{
                        if( value > 128 ) {
                            data[current]++;
                            if(data[current] > maxfeature){
                                j = max(j, k);
                                break;
                            }
                        }else{
                            current++;
                        }
                    }

                    if( current == 5 ){
                        const int size = VectorSize( data );
                        if( isFinder( size, data ) ){
                            retVal.push_back( Location( i, j + ( size / 2.0 ), size ) );
                            //j = max(j, k);
                            j += data[2] + data[3] + data[4];
                        }
                        
                        //j += 1;//data[0];// + data[1];
                        
                        break;
                    }
                }
            }
        }
    }

    return retVal;
}

const std::vector<Location> FindMarksVertical( Mat image, const int maxfeature ){
    std::vector<Location> retVal;

    for( unsigned int i = 0; i < image.cols; i++ ){
        for( unsigned int j = 0; j < image.rows; j++ ){
            if( image.at<unsigned char>(j, i) < 128 ) {
                int current = 0;
                std::vector<int> data( 5, 0 );

                for( unsigned int k = j - 1; k < image.rows; k++ ){
                    unsigned char value = image.at<unsigned char>(k, i);

                    if( current % 2 == 0 ) {
                        if( value < 128 ) {
                            data[current]++;
                            if(data[current] > maxfeature){
                                j = max(j, k);
                                break;
                            }
                        }else{
                            current++;
                        }
                    }else{
                        if( value > 128 ) {
                            data[current]++;
                            if(data[current] > maxfeature){
                                j = max(j, k);
                                break;
                            }
                        }else{
                            current++;
                        }
                    }

                    if( current == 5 ){
                        const int size = VectorSize( data );
                        if( isFinder( size, data ) ){
                            retVal.push_back( Location( j + ( size / 2.0 ), i , size) );
                            //j = max(j, k);
                            j += data[2] + data[3] + data[4];
                        }

                        //j += 1;//data[0];// + data[1];

                        break;
                    }
                }
            }
        }
    }

    return retVal;
}

void DrawCross( Mat image, const int row, const int column, const int size ){
    image.at<Vec3b>(row, column) = Red;

    // Draw Horizontal Line
    for( int i = 0; i < size; i ++){
        image.at<Vec3b>(row-i, column) = Red;
        image.at<Vec3b>(row+i, column) = Red;
    }

    // Draw Vertical Line
    for( int i = 0; i < size; i ++){
        image.at<Vec3b>(row, column-i) = Red;
        image.at<Vec3b>(row, column+i) = Red;
    }
}

int main( int argc, char **argv ) {
    // Confirm we have enough parameters
    if( argc != 2 ) {
        std::cerr << "ComputerVision2 <imagefile>" << std::endl;
        return -1;
    }

    // Read Image
    Mat image = imread( argv[1], 1 );
    
    const int cols = image.cols;
    const int rows = image.rows;
    
    float scale = 1.0;

    if( std::abs(cols - 1950) > 200){
        scale = (float)cols / 1950;
        const float inv_scale = 1.0 / scale;
        const int newrows = (int)(rows * inv_scale + 0.5);
    
        resize(image, image, Size(1950, newrows), 0, 0);
    }

    //std::cout << "Scale " << scale << std::endl;
    
    //pixels per feature
    const int halfpixelsperfeature = max(((image.rows / 21 / 7) + (image.cols / 28 / 7)) / 4, 1);
    const int maxfeature = halfpixelsperfeature * 2 * 3 * 2;
    
    Mat imblur;
    blur( image, imblur, Size( halfpixelsperfeature/2, halfpixelsperfeature/2 ), Point(-1,-1) );

    // Convert to Grayscale
    Mat gray;
    cvtColor(imblur, gray, CV_BGR2GRAY);

    // Threshold Image
    Mat temp;
    adaptiveThreshold( gray, temp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 127, 10 );//127, ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C

    // Test horizontally and vertically
    const std::vector<Location> Horizontal = FindMarksHorizontal( temp, maxfeature );
    const std::vector<Location> Vertical = FindMarksVertical( temp, maxfeature );

    // Find Centers
    std::vector<Location> locs;
    for( int i = 0; i < Horizontal.size(); i++ ){
        for( int j = 0; j < Vertical.size(); j++ ){
            if( Horizontal[i] == Vertical[j] ){
                const int locsize = locs.size();
                bool newpoint = true;
                for( int k=0; k<locsize; k++){
                    const Location loc = locs[k];
                    if(std::abs(loc.X - Horizontal[i].X) < 5 && std::abs(loc.Y - Horizontal[i].Y) < 5){
                        newpoint = false;
                        break;
                    }
                }
                
                if(newpoint){
                    std::cout << "Point: " << (int)(Horizontal[i].X * scale + 0.5) << ", " << (int)(Horizontal[i].Y * scale + 0.5) << ", " << (int)(((Horizontal[i].size + Vertical[j].size) / 2) * scale + 0.5)<< std::endl;

                    DrawCross( image, Horizontal[i].X, Horizontal[i].Y, 64 );
                    
                    locs.push_back(Horizontal[i]);
                }
                
            }
        }
    }
    
    // Draw to screen
    //cv::namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
    //cv::imshow( "Display Image", image );

    //while( cv::waitKey( 1 ) != 'q' ) {}

    cv::imwrite( "result.jpg", image );
    cv::imwrite( "grey.jpg", temp );

    return 0;
}
