# Self-Driving Car Engineer
Term 3: Path Planning Project

## Model Documentation
### General framework
The basis of my project code depends on the code shown in the project walkthrough. Therefore I use the spline library to get a smooth prediction of my current and future path. I enhanced the boilerplate code by implementing a lane change logic, that aims at driving both safe and efficient in the simulator. In general I could achieve a more or less smooth ride with my attempt but there are still some situations, where the ego car behaves unnaturaly. Especially when lane changes are not possible but would be necessary, the vehicle changes between slowing down and accelerating too aggressively.

A video of one of the final drives can be found here:
[![Alt text](https://img.youtube.com/vi/Q8ZLq4Zh4tY/0.jpg)](https://www.youtube.com/watch?v=Q8ZLq4Zh4tY)

### Lane change logic
I didn't calculate lane change costs as a function, but rather implemented a set of rules (ordered by priority) the ego car should follow. This implicitly apprixmiates my cost function and works as the bahaviour planner of the ego car.

First backward and forward distance of cars in neighouring lanes are checked. If the lane is not safe to drive, I mark this accordingly.
```cpp
bool forward_dist = ((check_car_s > car_s) && (((check_car_s - car_s) > max_dist_front)==false));
bool back_dist = ((check_car_s <= car_s) && (((car_s-check_car_s) > 5)==false));

double ref_d = 2+4*lane+2;
int car_lane = ((d - 4)/4)+1;

if (forward_dist || back_dist){

    if(car_lane == lane - 1 )   {
        safe_left = false;
        //std::cout << "Lane change right not safe." << std::endl;
    }
    else if (car_lane == lane + 1){
        safe_right = false;
        //std::cout << "Lane change left not safe." << std::endl;
    }
}
```

If both lanes are safe to drive on, choose the one with potential for higher speed.
```cpp
if (safe_right && safe_left){
    if(car_lane == lane - 1 )   {
        if (check_speed < min_speed_left){
            min_speed_left = check_speed;
        }
    }
    else if (car_lane == lane + 1) {
        if (check_speed < min_speed_right) {
            min_speed_right = check_speed;
        }
    }
}
```
A lane change depends on multiple conditions, which are encoded in the following code. Descriptions of each rule can be found inline.

```cpp
// perform lange changes if possible
// lane changes depend on multiple conditions, with descending priority from top to bottom. allow_strong_acc is a boolean, that prevents strong acceleration in case no lane change is possible and the ego car is slowed down by a car in front

// if driving on lane one and both lane changes are safe, prefer the one with potential for higher speed
if (too_close && lane == 1 && safe_left && safe_right){
    if (min_speed_left < min_speed_right){
        lane += 1;
        allow_strong_acc = true;
    } else {
        lane -= 1;
        allow_strong_acc = true;
    }
} else if (too_close && safe_left && lane > 0){ // otherwise prefer left lane changes is possible.
    lane -= 1;
    allow_strong_acc = true;
} else if (too_close && safe_right && lane < 2){
    lane += 1;
    allow_strong_acc = true;
}
else if ((too_close)&&(ref_vel>target_speed)){

    if(ref_vel - .224 >= target_speed){ // slow down, if lane change is not possible and car in front of ego to slow
        ref_vel -= .224;
    } else {
        ref_vel = target_speed;
    }

    if (lane == 0 and safe_right){ // try to go back to middle lane
        lane = 1;
        allow_strong_acc = true;
    } else if (lane == 2 and safe_left){ // try to go back to middle lane
        lane = 1;
        allow_strong_acc = true;
    }

} else if((too_close)&&(ref_vel<=target_speed)){
    ref_vel = target_speed;
} else if (lane != 1 && ref_vel >= 49.4) { // go back to middle lane, whenever possible
    if (lane == 2 && safe_left){
        lane = 1;
        allow_strong_acc = true;
    } else if (lane == 0 && safe_right){
        lane = 1;
        allow_strong_acc = true;
    }
} else if (ref_vel < 49.5){
    if (allow_strong_acc){ // accelerate, if trying to keep lane or change lanes
        ref_vel += .424;
    } else {
        ref_vel += .224; // during "prepare lane change"
    }
}
```