/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <climits>
#include <random>
#include <unistd.h>
#include <omp.h>

void print_stats(const std::vector<std::vector<int>>& occupancy) {
  int max_occupancy = 0;
  long long total_cost = 0;

  for (const auto& row : occupancy) {
    for (const int count : row) {
      max_occupancy = std::max(max_occupancy, count);
      total_cost += count * count;
    }
  }

  std::cout << "Max occupancy: " << max_occupancy << '\n';
  std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(const std::vector<Wire>& wires, const int num_wires, const std::vector<std::vector<int>>& occupancy, const int dim_x, const int dim_y, const int num_threads, std::string input_filename) {
  if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
    input_filename.resize(std::size(input_filename) - 4);
  }

  const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(num_threads) + ".txt";
  const std::string wires_filename = input_filename + "_wires_" + std::to_string(num_threads) + ".txt";

  std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
  if (!out_occupancy) {
    std::cerr << "Unable to open file: " << occupancy_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_occupancy << dim_x << ' ' << dim_y << '\n';
  for (const auto& row : occupancy) {
    for (const int count : row) {
      out_occupancy << count << ' ';
    }
    out_occupancy << '\n';
  }

  out_occupancy.close();

  std::ofstream out_wires(wires_filename, std::fstream:: out);
  if (!out_wires) {
    std::cerr << "Unable to open file: " << wires_filename << '\n';
    exit(EXIT_FAILURE);
  }

  out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

  for (const auto& [start_x, start_y, end_x, end_y, bend1_x, bend1_y] : wires) {
    out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

    if (start_y == bend1_y) {
    // first bend was horizontal

      if (end_x != bend1_x) {
        // two bends

        out_wires << bend1_x << ' ' << end_y << ' ';
      }
    } else if (start_x == bend1_x) {
      // first bend was vertical

      if(end_y != bend1_y) {
        // two bends

        out_wires << end_x << ' ' << bend1_y << ' ';
      }
    }
    out_wires << end_x << ' ' << end_y << '\n';
  }

  out_wires.close();
}
int calcCost(std::vector<std::vector<int>>& occupancy ,int xi,int yi, int xf, int yf, int bendX, int bendY, int dim_x, int dim_y)
{ 

  int cost = 0;
  if(bendX == xf)
  {
    for(int i = std::min(yi,bendY); i < std::max(yi,bendY); i++)
    {
      if(i>=0 && i< dim_y)
        {
          cost += occupancy[xi][i] + 1;
        }
      
    }
    for(int i = std::min(xi,xf); i < std::max(xi,xf); i ++)
    {
      if(i>=0 && i< dim_x)
        {
      cost += occupancy[i][bendY] + 1;
        }
    }
    for(int i = std::min(bendY,yf); i <= std::max(bendY,yf); i ++)
    {
      if(i>=0 && i< dim_y)
        {
      cost += occupancy[xf][i] + 1;
        }

    }
  }
  else{
    for(int i = std::min(xi,bendX); i < std::max(xi,bendX); i++)
    {
      if(i>=0 && i< dim_x)
        {
      cost += occupancy[yi][i] + 1;
        }
    }
    for(int i = std::min(yi,yf); i < std::max(yi,yf); i ++)
    {
      if(i>=0 && i< dim_y)
        {
      cost += occupancy[bendX][i] + 1;
        }
    }
    for(int i = std::min(bendX,xf); i <= std::max(bendX,xf); i ++)
    {
      if(i>=0 && i< dim_x)
        {
      cost += occupancy[i][yf] + 1;
        }
    }
  }
  return cost;
  
}

void update(std::vector<std::vector<int>>& occupancy , struct Wire route, int dim_x, int dim_y)
{         

          int bendX = route.bend1_x;
          int bendY = route.bend1_y;
          int xi = route.start_x;
          int yi = route.start_y;
          int xf = route.end_x;
          int yf = route.end_y;
          
          if(bendX == xf)
          {
            for(int i = std::min(yi,bendY); i < std::max(yi,bendY); i++)
            {
              if(i>=0 && i< dim_x)
              {
                occupancy[i][xi] += 1;
              }
              
            }
            for(int i = std::min(xi,xf); i < std::max(xi,xf); i ++)
            {
              if(i>=0 && i < dim_y)
              {
              occupancy[bendY][i] += 1;
              }
            }
            for(int i = std::min(bendY,yf); i <= std::max(bendY,yf); i ++)
            {
              if(i>=0 && i< dim_x)
              {
              occupancy[i][xf] += 1;
              }

            }
          }
          else{
            for(int i = std::min(xi,bendX); i < std::max(xi,bendX); i++)
            {
              if(i>=0 && i< dim_y)
              {
              occupancy[yi][i] += 1;
              }
            }
            for(int i = std::min(yi,yf); i < std::max(yi,yf); i ++)
            {
              if(i>=0 && i< dim_x)
              {
              occupancy[i][bendX] += 1;
              }
            }
            for(int i = std::min(bendX,xf); i <= std::max(bendX,xf); i ++)
            {
              if(i>=0 && i< dim_y)
              {
              occupancy[yf][i] += 1;
              }
            }
          }

}

void removeInitial(std::vector<std::vector<int>>& occupancy , struct Wire route, int dim_x, int dim_y)
{
  int bendX = route.bend1_x;
          int bendY = route.bend1_y;
          int xi = route.start_x;
          int yi = route.start_y;
          int xf = route.end_x;
          int yf = route.end_y;
          
          if(bendX == xf)
          {
            for(int i = std::min(yi,bendY); i < std::max(yi,bendY); i++)
            {
              if(i>=0 && i< dim_x)
              {
                occupancy[i][xi] -= 1;
              }
              
            }
            for(int i = std::min(xi,xf); i < std::max(xi,xf); i ++)
            {
              if(i>=0 && i < dim_y)
              {
              occupancy[bendY][i] -= 1;
              }
            }
            for(int i = std::min(bendY,yf); i <= std::max(bendY,yf); i ++)
            {
              if(i>=0 && i< dim_x)
              {
              occupancy[i][xf] -= 1;
              }

            }
          }
          else{
            for(int i = std::min(xi,bendX); i < std::max(xi,bendX); i++)
            {
              if(i>=0 && i< dim_y)
              {
              occupancy[yi][i] -= 1;
              }
            }
            for(int i = std::min(yi,yf); i < std::max(yi,yf); i ++)
            {
              if(i>=0 && i< dim_x)
              {
              occupancy[i][bendX] -= 1;
              }
            }
            for(int i = std::min(bendX,xf); i <= std::max(bendX,xf); i ++)
            {
              if(i>=0 && i< dim_y)
              {
              occupancy[yf][i] -= 1;
              }
            }
          }
}

int refOccupancy(std::vector<std::vector<int>>& occupancy , struct Wire route, int dim_x, int dim_y, int flag){
  // If flag == -1, decrement occupancy along route
  // If flag == 1, increment occupancy along route
  // If flag == 0, calculate cost of adding the route
  int bend2_x;
  int bend2_y;
  int start_x = route.start_x;
  int start_y = route.start_y;
  int end_x = route.end_x;
  int end_y = route.end_y;
  int bend1_x = route.bend1_x;
  int bend1_y = route.bend1_y;

  if (bend1_x == start_x) {
    bend2_x = end_x;
    bend2_y = bend1_y;
  }
  else if (bend1_y == start_y) {
    bend2_x = bend1_x;
    bend2_y = end_y;
  }
  else {
    printf("Should not have got here!\n");
    return -109823498;
  }


  int cost = 0;

  // START TO BEND 1
  int stepi1 = 1;
  if(start_y > bend1_y)
  {
    stepi1 = -1;
  }
  for (int i = start_y ; i != bend1_y; i += stepi1){
    // if (i == 3) printf("bruh\n");
    if (flag == 0){
      cost += occupancy[i][start_x] + 1;
    }
    else {
      occupancy[i][start_x] += flag;
    }
  }
  int stepi2 = 1;
  if(start_x > bend1_x)
  {
    stepi2 = -1;
  }
  for (int i = start_x; i != bend1_x; i += stepi2 ) {
    if (flag == 0){
      cost += occupancy[start_y][i] + 1;
    }
    else {
      occupancy[start_y][i] += flag;
    }
  }
  // printf("START = (%d, %d)\n", start_x, start_y);
  // printf("BEND1 = (%d, %d)\n", bend1_x, bend1_y);
  // printf("BEND2 = (%d, %d)\n", bend2_x, bend2_y);
  // printf("END = (%d, %d)\n", end_x, end_y);


  int stepi3 = 1;
  if(bend1_x > bend2_x)
  {
    stepi3 = -1;
  }
  // BEND 1 TO BEND 2
  for (int i = bend1_x; i !=  bend2_x; i += stepi3) {
    
    if (flag == 0){
      cost += occupancy[bend1_y][i] + 1;
    }
    else {
      occupancy[bend1_y][i] += flag;
    }
  }

  int stepi4 = 1;
  if(bend1_y > bend2_y)
  {
    stepi4 = -1;
  }
  
  for (int i = bend1_y; i !=  bend2_y; i += stepi4) {
    
    if (flag == 0){
      cost += occupancy[i][bend1_x] + 1;
    }
    else {
      occupancy[i][bend1_x] += flag;
    }
  }

  int stepi5 = 1;
  if(bend2_x > end_x)
  {
    stepi5 = -1;
  }


  // BEND 2 TO END
  for (int i = bend2_x ; i != end_x; i += stepi5) {
    if (flag == 0){
      cost += occupancy[end_y][i] + 1;
    }
    else {
      occupancy[end_y][i] += flag;
    }
  }

  int stepi6 = 1;
  if(bend2_y > end_y)
  {
    stepi6 = -1;
  }
  for (int i = bend2_y; i !=  end_y; i +=stepi6) {
    
    if (flag == 0){
      cost += occupancy[i][end_x] + 1;
    }
    else {
      occupancy[i][end_x] += flag;
    }
  }

  // INCLUDE END POINT
  if (flag == 0){
      cost += occupancy[end_y][end_x] + 1;
    }
  else {
    occupancy[end_y][end_x] += flag;
  }
  return cost;

}


int main(int argc, char *argv[]) {
  const auto init_start = std::chrono::steady_clock::now();

  std::string input_filename;
  int num_threads = 0;
  double SA_prob = 0.1;
  int SA_iters = 5;
  char parallel_mode = '\0';
  int batch_size = 1;

  int opt;
  while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
    switch (opt) {
      case 'f':
        input_filename = optarg;
        break;
      case 'n':
        num_threads = atoi(optarg);
        break;
      case 'p':
        SA_prob = atof(optarg);
        break;
      case 'i':
        SA_iters = atoi(optarg);
        break;
      case 'm':
        parallel_mode = *optarg;
        break;
      case 'b':
        batch_size = atoi(optarg);
        break;
      default:
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        exit(EXIT_FAILURE);
    }
  }

  // Check if required options are provided
  if (empty(input_filename) || num_threads <= 0 || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
    std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
    exit(EXIT_FAILURE);
  }

  std::cout << "Number of threads: " << num_threads << '\n';
  std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
  std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
  std::cout << "Input file: " << input_filename << '\n';
  std::cout << "Parallel mode: " << parallel_mode << '\n';
  std::cout << "Batch size: " << batch_size << '\n';

  std::ifstream fin(input_filename);

  if (!fin) {
    std::cerr << "Unable to open file: " << input_filename << ".\n";
    exit(EXIT_FAILURE);
  }

  int dim_x, dim_y;
  int num_wires;

  /* Read the grid dimension and wire information from file */
  fin >> dim_x >> dim_y >> num_wires;

  std::vector<Wire> wires(num_wires);
  //Changed this line below (bugs)
  std::vector<std::vector<int>> occupancy(dim_y, std::vector<int>(dim_x)); 

  for (auto& wire : wires) {
    fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
    wire.bend1_x = wire.start_x;
    wire.bend1_y = wire.start_y;
  }

  /* Initialize any additional data structures needed in the algorithm */

  const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
  std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

  const auto compute_start = std::chrono::steady_clock::now();

  /** 
   * Implement the wire routing algorithm here
   * Feel free to structure the algorithm into different functions
   * Don't use global variables.
   * Use OpenMP to parallelize the algorithm. 
   */

  if(parallel_mode == 'W')
  {
    for(int iter = 0; iter < SA_iters; iter++)
    {
      //On first iteration, place initial routes of wire (very naive ask NOOR)
      if(iter == 0)
      {
        for(int wireIndex = 0; wireIndex < num_wires; wireIndex++)
        {
          struct Wire currWire = wires[wireIndex];
          currWire.bend1_x = currWire.start_x;
          currWire.bend1_y = currWire.end_y;
          wires[wireIndex] = currWire;
          // printf("INITIAL COST = %d",refOccupancy(occupancy, currWire,  dim_x,  dim_y, 0) );
          refOccupancy(occupancy, currWire,  dim_x,  dim_y, 1);
          // printf("INITIAL COST = %d",refOccupancy(occupancy, currWire,  dim_x,  dim_y, 0) );

          
        }
      }
      else
      {
      //Iterate over each wire sequentially
        for(int wireIndex = 0; wireIndex < num_wires; wireIndex++)
          {
            struct Wire currWire = wires[wireIndex];
            int xi, yi, xf, yf;
            xi = currWire.start_x;
            yi = currWire.start_y;
            xf = currWire.end_x;
            yf = currWire.end_y;
            // int bendx_init = currWire.bend1_x;
            // int bendy_init = currWire.bend1_y;
            int delta_x = std::abs(xf - xi);
            int delta_y = std::abs(yf - yi);
            if(delta_x == 0 || delta_y == 0 )
            {
              continue;
            }
            //Check straight line wire
            // if(delta_x == 0)
            // {
            //   //Only update straight line wire on first iteration
            //   if( iter == 0){
            //     for(int i = std::min(yi,yf); i <= std::max(yi,yf); i++ )
            //     {
            //       occupancy[i][xf]+=1;
            //     }
            //   }
            //   //Skip rest of loop body
            //   else
            //   {
            //     continue;
            //   }
            // }
            
            // else if (delta_y == 0)
            // {
            //   if(iter == 0)
            //   {

            //     for(int i = std::min(xi,xf); i <= std::max(xi,xf); i++ )
            //     {
            //       occupancy[yi][i]+=1;
            //     }
            //   }
            //   else
            //   {
            //     continue;
            //   }
            // }
            
            else
            {
              refOccupancy(occupancy,currWire,dim_x,dim_y, -1);
              int initial_cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
              int* costs = (int*) malloc(sizeof(int) * (delta_x + delta_y));
              struct Wire* possRoutes = (struct Wire*)malloc(sizeof(struct Wire)*(delta_x + delta_y));
              int chunk_size; 
              
              omp_set_num_threads(num_threads);
              // #pragma omp parallel
              // {
                chunk_size = 1;
                #pragma omp parallel for schedule(static, chunk_size)
                for (int threadId = 0; threadId < delta_x; threadId ++)
                {
                  // int bendX;
                  // int bendY;
                  // bendY = yi; // bendY = yi
                  // bendX = std::min(xi,xf) + threadId + 1;
                  if(xi > xf)
                  {
                    currWire.bend1_x = xi - threadId - 1;
                  }
                  else {
                    currWire.bend1_x = xi + threadId + 1;
                  }
                  
                  currWire.bend1_y = yi;

                  int cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
                  costs[threadId] = cost;
                  possRoutes[threadId] = currWire;
                  
                  // atomic check set min
                }
              
              chunk_size = 1;
              #pragma omp parallel for schedule(static, chunk_size)
              
                for (int threadId = 0; threadId <  delta_y; threadId ++)
                  {
                    // int bendX;
                    // int bendY;
                  
                    // bendX = xi; // bendX =xi
                    // bendY = std::min(yi,yf) + threadId + 1;
                    currWire.bend1_x = xi;
                    if (yi > yf) {
                      currWire.bend1_y = yi - threadId - 1;
                    }
                    else {
                      currWire.bend1_y = yi + threadId + 1;
                    }
                    
      
                    int cost = refOccupancy(occupancy, currWire, dim_x, dim_y, 0);
                    costs[delta_x + threadId] = cost;
                    possRoutes[delta_x + threadId] = currWire;
                    
                    // atomic check set min
                  }
              
            // }
          //#pragma omp barrier

              int min_ind = 0;
              int min_cost = INT_MAX;
              std::random_device rd;  // obtain a random number from hardware
              std::mt19937 gen(rd()); // seed the generator

              // Define a distribution (uniform distribution between 0 and 1)
              std::uniform_real_distribution<> dis(0.0, 1.0);

              // Generate a random number between 0 and 1
              float random_number = dis(gen);
              
              if(random_number < SA_prob)
              {
                std::uniform_int_distribution<> dis(0, delta_x + delta_y - 1);
                int random_index= dis(gen);
                // currWire.bend1_x = bendx_init;
                // currWire.bend1_y = bendy_init;
                //struct Wire original_wire = wires[wireIndex];
                //refOccupancy(occupancy,original_wire,dim_x,dim_y, -1);
                refOccupancy(occupancy, possRoutes[random_index],  dim_x,  dim_y, 1);
                //update wire
                wires[wireIndex] = possRoutes[random_index];
              }
              else
              {
                for (int i = 0; i < delta_x + delta_y; i ++) 
                {
                  // printf("Iter: %d, Wire: %d, Cost: %d\n",iter,wireIndex,costs[i]);
                  if (costs[i] < min_cost)
                  {
                    min_cost = costs[i];
                    min_ind = i;
                  }

                }
                struct Wire original_wire = wires[wireIndex];
                if (min_cost < initial_cost) 
                {
                  // currWire.bend1_x = bendx_init;
                  // currWire.bend1_y = bendy_init;
                  // printf("min cost is  %d and initial is %d \n", min_cost, initial_cost);
                  
                  refOccupancy(occupancy, possRoutes[min_ind], dim_x,  dim_y, 1);
                  wires[wireIndex] = possRoutes[min_ind];
                }
                else{
                    refOccupancy(occupancy,original_wire,dim_x,dim_y, 1);

                }
              }
            }
          }
        }
      }
    }

    //Parallelism between wires
    else
    {
      //dynamic omp synch, locks, batches
    } 

  const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
  std::cout << "Computation time (sec): " << compute_time << '\n';

  /* Write wires and occupancy matrix to files */

  print_stats(occupancy);
  write_output(wires, num_wires, occupancy, dim_x, dim_y, num_threads, input_filename);
}

validate_wire_t Wire::to_validate_format(void) const {
  /* TODO(student): Implement this if you want to use the wr_checker. */
  /* See wireroute.h for details on validate_wire_t. */
  throw std::logic_error("to_validate_format not implemented.");
}
