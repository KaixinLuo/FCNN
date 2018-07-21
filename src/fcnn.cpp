#include <iostream>
#include <cmath>
class ActivationFunction{
    protected:
        float input;
        virtual float operator () (float input);
        virtual float diff(float input);
};

class Sigmoid:public ActivationFunction{
    public:
        Sigmoid();
        ~Sigmoid();
        float operator()(float input);
        float diff(float input);
};
Sigmoid::Sigmoid(){
    ;
}
Sigmoid::~Sigmoid(){
    ;
}
float Sigmoid::operator()(float input){
    this->input = input;
    return (1)/(1+exp(-input));
}
float Sigmoid::diff(float input){
    return (exp(-input))/((1+exp(-input))*(1+exp(-input)));
}

class ReLU:public ActivationFunction{
    public:
        ReLU();
        ~ReLU();
        float operator()(float input);
        float diff(float input);
};
ReLU::ReLU(){
    ;
}
ReLU::~ReLU(){
    ;
}
float ReLU::operator()(float input){
    this->input = input;
    return (input>0)?input:0;
}
float ReLU::diff(float input){
    return (input>0)?1:0;
}

class AbstractNeuron{
    protected:
        virtual float operator()(float *input);
        virtual float operator[] (int index);
        virtual float calculatePartialError(float error,int index);
        virtual void calculateDelta(float error);
        virtual void applyUpdate(float learningRate);
};
class AbstractLayer{
    protected:
        ActivationFunction f;
        int numOfInput;
        int ID;
};

class NormalNeuron:public AbstractNeuron{
    private:
        float *input;
        float delta;
        float *weights;
        float bias;
        int ID;

        AbstractLayer host;
    public:
        NormalNeuron(AbstractLayer layer, int ID);
        ~NormalNeuron();
        float operator()(float *input);
        float operator[](int index);
        float calculatePartialError(float error,int index);
        void calculateDelta(float error);
        void applyUpdate(float learningRate);
};
