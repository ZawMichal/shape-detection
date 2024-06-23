using System;
using System.IO;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;


namespace shape_detection
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            LoadImageOnStart();
        }
        private void LoadImageOnStart()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Title = "Select an image",
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif"
            };

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                string filePath = openFileDialog.FileName;
                try
                {
                    // Load the image
                    Mat mat = Cv2.ImRead(filePath, ImreadModes.Color);

                    // Display original image in pictureBox1
                    Bitmap originalBitmap = BitmapConverter.ToBitmap(mat);
                    pictureBox1.Image = originalBitmap;
                    pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;

                    // Process image to detect and frame shapes
                    Bitmap contourBitmap = DrawContours(mat);
                    pictureBox2.Image = contourBitmap;
                    pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;

                    // Clean up resources
                    mat.Dispose();
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error loading image: " + ex.Message);
                }
            }
        }

        private Bitmap DrawContours(Mat inputMat)
        {
            // Convert to grayscale
            Mat grayMat = new Mat();
            Cv2.CvtColor(inputMat, grayMat, ColorConversionCodes.BGR2GRAY);

            // Apply GaussianBlur to reduce noise
            Cv2.GaussianBlur(grayMat, grayMat, new OpenCvSharp.Size(5, 5), 0);

            // Apply Canny edge detection
            Mat edges = new Mat();
            Cv2.Canny(grayMat, edges, 50, 150);

            // Find contours
            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(edges, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            // Create a new Mat for drawing result
            Mat resultMat = inputMat.Clone();

            // List to store information about shapes
            List<(OpenCvSharp.Point[], string)> shapesInfo = new List<(OpenCvSharp.Point[], string)>();

            // Process each contour
            foreach (var contour in contours)
            {
                // Approximate contour to polygon
                OpenCvSharp.Point[] approxCurve = Cv2.ApproxPolyDP(contour, Cv2.ArcLength(contour, true) * 0.02, true);

                // Determine the shape type
                int vertices = approxCurve.Length;

                string shapeName = GetShapeName(vertices);

                // Add shape information to the list
                shapesInfo.Add((contour, shapeName));
            }

            // Draw shapes based on collected information
            foreach (var (contour, shapeName) in shapesInfo)
            {
                DrawBoundingRect(resultMat, contour, shapeName);
            }

            // Convert Mat back to Bitmap
            Bitmap resultBitmap = BitmapConverter.ToBitmap(resultMat);

            // Clean up resources
            grayMat.Dispose();
            edges.Dispose();
            resultMat.Dispose();

            return resultBitmap;
        }

        private string GetShapeName(int vertices)
        {
            string shapeName = "";
            switch (vertices)
            {
                case 3:
                    shapeName = "Triangle";
                    break;
                case 4:
                    shapeName = "Rectangle";
                    break;
                case 5:
                    shapeName = "Pentagon";
                    break;
                default:
                    shapeName = "Circle";
                    break;
            }
            return shapeName;
        }

        private void DrawBoundingRect(Mat image, OpenCvSharp.Point[] contour, string shapeName)
        {
            // Find bounding rectangle
            RotatedRect rect = Cv2.MinAreaRect(contour);
            OpenCvSharp.Point2f[] vertices = rect.Points();

            // Convert Point2f to Point
            OpenCvSharp.Point[] points = Array.ConvertAll(vertices, PointFromPointF);

            // Expand the bounding rectangle
            OpenCvSharp.Rect expandedRect = ExpandRect(rect.BoundingRect(), 5); // Expand by 5 pixels

            // Draw expanded bounding rectangle on image
            Cv2.Rectangle(image, expandedRect, Scalar.Red, 2);

            // Create a mask for the contour
            Mat mask = new Mat(image.Rows, image.Cols, MatType.CV_8UC1, Scalar.Black);
            Cv2.FillConvexPoly(mask, contour, Scalar.White);

            // Find mean color inside the contour using the mask
            Scalar meanColor = Cv2.Mean(image, mask);

            // Determine color name based on mean color
            string colorName = GetColorName(meanColor);

            // Combine shape name with color name
            string labeledShapeName = $"{colorName} {shapeName}";

            // Define the position for the shape label text
            OpenCvSharp.Point textPos = new OpenCvSharp.Point(expandedRect.Left, expandedRect.Bottom + 20); // Below the rectangle

            // Draw labeled shape text
            Cv2.PutText(image, labeledShapeName, textPos, HersheyFonts.HersheyComplex, 1.0, Scalar.Blue, 2);
        }


        private string GetColorName(Scalar color)
        {
            // Define ranges for different colors (you can adjust these ranges)
            Scalar yellowMin = new Scalar(20, 100, 100);
            Scalar yellowMax = new Scalar(40, 255, 255);
            Scalar redMin = new Scalar(0, 0, 100);
            Scalar redMax = new Scalar(40, 40, 255);
            Scalar blueMin = new Scalar(100, 0, 0);
            Scalar blueMax = new Scalar(255, 40, 40);
            Scalar lightBlueMin = new Scalar(100, 100, 0);
            Scalar lightBlueMax = new Scalar(255, 255, 40);
            Scalar darkBlueMin = new Scalar(0, 0, 80);
            Scalar darkBlueMax = new Scalar(100, 100, 255); // Adjusted range for dark blue
            Scalar greenMin = new Scalar(0, 100, 0);
            Scalar greenMax = new Scalar(40, 255, 40);
            Scalar lightGreenMin = new Scalar(0, 150, 0);
            Scalar lightGreenMax = new Scalar(40, 255, 40);
            Scalar darkGreenMin = new Scalar(0, 50, 0);
            Scalar darkGreenMax = new Scalar(60, 150, 60); // Adjusted range for dark green
            Scalar orangeMin = new Scalar(0, 70, 150);
            Scalar orangeMax = new Scalar(40, 190, 255);
            Scalar purpleMin = new Scalar(80, 0, 80);
            Scalar purpleMax = new Scalar(255, 80, 255);
            Scalar pinkMin = new Scalar(150, 0, 100);
            Scalar pinkMax = new Scalar(255, 40, 255);
            Scalar brownMin = new Scalar(0, 50, 50);
            Scalar brownMax = new Scalar(40, 150, 150);
            Scalar blackMin = new Scalar(0, 0, 0);
            Scalar blackMax = new Scalar(40, 40, 40);

            // Check if the color falls within each range
            if (IsInRange(color, yellowMin, yellowMax))
                return "Yellow";
            else if (IsInRange(color, redMin, redMax))
                return "Red";
            else if (IsInRange(color, blueMin, blueMax))
                return "Blue";
            else if (IsInRange(color, lightBlueMin, lightBlueMax))
                return "Light Blue";
            else if (IsInRange(color, darkBlueMin, darkBlueMax))
                return "Dark Blue";
            else if (IsInRange(color, greenMin, greenMax))
                return "Green";
            else if (IsInRange(color, lightGreenMin, lightGreenMax))
                return "Light Green";
            else if (IsInRange(color, darkGreenMin, darkGreenMax))
                return "Dark Green";
            else if (IsInRange(color, orangeMin, orangeMax))
                return "Orange";
            else if (IsInRange(color, purpleMin, purpleMax))
                return "Purple";
            else if (IsInRange(color, pinkMin, pinkMax))
                return "Pink";
            else if (IsInRange(color, brownMin, brownMax))
                return "Brown";
            else if (IsInRange(color, blackMin, blackMax))
                return "Black";
            else
                return "Unknown";
        }

        private bool IsInRange(Scalar color, Scalar min, Scalar max)
        {
            return color.Val0 >= min.Val0 && color.Val0 <= max.Val0 &&
                   color.Val1 >= min.Val1 && color.Val1 <= max.Val1 &&
                   color.Val2 >= min.Val2 && color.Val2 <= max.Val2;
        }


        private OpenCvSharp.Point PointFromPointF(OpenCvSharp.Point2f point2f)
        {
            return new OpenCvSharp.Point((int)point2f.X, (int)point2f.Y);
        }

        private OpenCvSharp.Rect ExpandRect(OpenCvSharp.Rect rect, int expandBy)
        {
            return new OpenCvSharp.Rect(
                Math.Max(0, rect.Left - expandBy),
                Math.Max(0, rect.Top - expandBy),
                rect.Width + 2 * expandBy,
                rect.Height + 2 * expandBy);
        }
    }
}
