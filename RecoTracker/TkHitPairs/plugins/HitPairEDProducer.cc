#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RunningAverage.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionsSeedingLayerSets.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/graph/default_device.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "TH2F.h"

#include <chrono>

// #include <algorithm>
// #include <chrono>
// #include <cstdlib>
// #include <cuda_runtime_api.h>
// #include <fstream>
// #include <iostream>
// #include <string>
// #include <sys/stat.h>
// #include <unordered_map>
// #include <cassert>
// #include <vector>
// #include "NvInfer.h"
// #include "NvUffParser.h"
//
// #include "NvUtils.h"

namespace { class ImplBase; }

class HitPairEDProducer: public edm::stream::EDProducer<> {
public:
  HitPairEDProducer(const edm::ParameterSet& iConfig);
  ~HitPairEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<bool> clusterCheckToken_;

  std::unique_ptr<::ImplBase> impl_;
};

namespace {
  class ImplBase {
  public:
    ImplBase(const edm::ParameterSet& iConfig);
    virtual ~ImplBase() = default;

    virtual void produces(edm::ProducerBase& producer) const = 0;

    virtual void produce(const bool clusterCheckOk, edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  protected:
    edm::RunningAverage localRA_;
    const unsigned int maxElement_;

    bool doInference_;
    float t_;

    HitPairGeneratorFromLayerPair generator_;
    std::vector<unsigned> layerPairBegins_;
  };
  ImplBase::ImplBase(const edm::ParameterSet& iConfig):
    maxElement_(iConfig.getParameter<unsigned int>("maxElement")),
    doInference_(iConfig.existsAs<bool>("doInference") ? iConfig.getParameter<bool>("doInference") : true),
    t_(iConfig.existsAs<double>("thresh") ? iConfig.getParameter<double>("thresh") : 0.1),
    generator_(0, 1, nullptr, maxElement_), // these indices are dummy, TODO: cleanup HitPairGeneratorFromLayerPair
    layerPairBegins_(iConfig.getParameter<std::vector<unsigned> >("layerPairs"))
  {
    if(layerPairBegins_.empty())
      throw cms::Exception("Configuration") << "HitPairEDProducer requires at least index for layer pairs (layerPairs parameter), none was given";
  }

  /////
  template <typename T_SeedingHitSets, typename T_IntermediateHitDoublets, typename T_RegionLayers>
  class Impl: public ImplBase {
  public:
    template <typename... Args>
    Impl(const edm::ParameterSet& iConfig, Args&&... args):
      ImplBase(iConfig),
      regionsLayers_(&layerPairBegins_, std::forward<Args>(args)...)
    {}
    ~Impl() override = default;

    void produces(edm::ProducerBase& producer) const override {
      T_SeedingHitSets::produces(producer);
      T_IntermediateHitDoublets::produces(producer);
    }

    // HitDoublets cnnInference( const TrackingRegion& region,
    //                           const edm::EventSetup& es,
    //                           HitDoublets& copyDoublets,SeedingLayerSetsHits::SeedingLayerSet layerSet,
    //                           LayerHitMapCache & layerCache) const
    HitDoublets cnnInference(HitDoublets& thisDoublets)
    {
      // const RecHitsSortedInPhi & innerHitsMap = layerCache(layerSet[0], region, es);
      // const RecHitsSortedInPhi& outerHitsMap = layerCache(layerSet[1], region, es);
      //
      // HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));

      auto startData = std::chrono::high_resolution_clock::now();

      std::vector< float > inPad, outPad;

      // Load graph
      tensorflow::setLogging("0");
      std::cout << "GPU" << std::endl;
      std::vector<int> pixelDets{0,1,2,3,14,15,16,29,30,31}, layerIds;

      tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/lustre/home/adrianodif/CNNDoublets/freeze_models/dense_pix_model_final.pb");
      tensorflow::Session* session = tensorflow::createSession(graphDef,16);

      for (int i = 0; i < graphDef->node_size(); ++i)
      {
        auto node = graphDef->mutable_node(i);
        if (node->device().empty()) {
          node->set_device("/device:GPU:0");
        }
      }

      int numOfDoublets = thisDoublets.size(), padSize = 16, cnnLayers = 10, infoSize = 67;
      float padHalfSize = 8.0;
      // tensorflow::Tensor inputPads(tensorflow::DT_FLOAT, {numOfDoublets,padSize,padSize,cnnLayers*2});
      tensorflow::Tensor inputFeat(tensorflow::DT_FLOAT, {numOfDoublets,infoSize});

      // float* vPad = inputPads.flat<float>().data();
      float* vLab = inputFeat.flat<float>().data();

      HitDoublets copyDoublets = std::move(thisDoublets);

      // std::cout << "copyDoublets.size()=" << copyDoublets.size() << std::endl;
      // std::cout << "thisDoublets.size()=" << thisDoublets.size() << std::endl;

      //return copyDoublets;

      DetLayer const * innerLayer = copyDoublets.detLayer(HitDoublets::inner);
      // if(find(pixelDets.begin(),pixelDets.end(),innerLayer->seqNum())==pixelDets.end()) return copyDoublets;

      DetLayer const * outerLayer = copyDoublets.detLayer(HitDoublets::outer);
      // if(find(pixelDets.begin(),pixelDets.end(),outerLayer->seqNum())==pixelDets.end()) return copyDoublets;

      std::vector <unsigned int> detSeqs;
      detSeqs.push_back(innerLayer->seqNum());
      detSeqs.push_back(outerLayer->seqNum());

      layerIds.push_back(find(pixelDets.begin(),pixelDets.end(),innerLayer->seqNum()) - pixelDets.begin());
      layerIds.push_back(find(pixelDets.begin(),pixelDets.end(),outerLayer->seqNum()) - pixelDets.begin());
      std::vector<tensorflow::Tensor> outputs;

      HitDoublets::layer layers[2] = {HitDoublets::inner, HitDoublets::outer};

      std::vector < float > zeroPad;
      for (int nx = 0; nx < padSize; ++nx)
        for (int ny = 0; ny < padSize; ++ny)
          zeroPad.push_back(0.0);

      std::vector<int> inIndex, outIndex;
      for (int iD = 0; iD < numOfDoublets; iD++)
      {

        //copyDoublets.add()
        std::vector <unsigned int> subDetIds, detIds ;

        // std::vector< std::vector< float>> hitPads,inHitPads,outHitPads;

        // for(int i = 0; i < cnnLayers; ++i)
        // {
        //   inHitPads.push_back(zeroPad);
        //   outHitPads.push_back(zeroPad);
        // }

        float deltaA = 0.0, deltaADC = 0.0, deltaS = 0.0, deltaR = 0.0;
        float deltaPhi = 0.0, deltaZ = 0.0, zZero = 0.0;

        int iLab = 0;
        int doubOffset = (padSize*padSize*cnnLayers*2)*iD, infoOffset = (infoSize)*iD;

        std::vector< RecHitsSortedInPhi::Hit> hits;
        std::vector< const SiPixelRecHit*> siHits;

        siHits.push_back(dynamic_cast<const SiPixelRecHit*>(copyDoublets.hit(iD, HitDoublets::inner)->hit()));
        siHits.push_back(dynamic_cast<const SiPixelRecHit*>(copyDoublets.hit(iD, HitDoublets::outer)->hit()));

        detIds.push_back(copyDoublets.hit(iD, HitDoublets::inner)->hit()->geographicalId());
        subDetIds.push_back((copyDoublets.hit(iD, HitDoublets::inner)->hit()->geographicalId()).subdetId());

        inIndex.push_back(copyDoublets.index(iD,HitDoublets::inner));
        outIndex.push_back(copyDoublets.index(iD,HitDoublets::outer));

        detIds.push_back(copyDoublets.hit(iD, HitDoublets::outer)->hit()->geographicalId());
        subDetIds.push_back((copyDoublets.hit(iD, HitDoublets::outer)->hit()->geographicalId()).subdetId());

        if (! (((subDetIds[0]==1) || (subDetIds[0]==2)) && ((subDetIds[1]==1) || (subDetIds[1]==2)))) continue;
        //
        // hitPads.push_back(inPad);
        // hitPads.push_back(outPad);

        for(int j = 0; j < 2; ++j)
        {

          int padOffset = layerIds[j] * padSize * padSize + j * padSize * padSize * cnnLayers;

          vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.x(); iLab++;
          vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.y(); iLab++;
          vLab[iLab + infoOffset] = (float)(siHits[j]->globalState()).position.z(); iLab++;

          float phi = copyDoublets.phi(iD,layers[j]) >=0.0 ? copyDoublets.phi(iD,layers[j]) : 2*M_PI + copyDoublets.phi(iD,layers[j]);
          vLab[iLab + infoOffset] = (float)phi; iLab++;
          vLab[iLab + infoOffset] = (float)copyDoublets.r(iD,layers[j]); iLab++;

          vLab[iLab + infoOffset] = (float)detSeqs[j]; iLab++;

          if(subDetIds[j]==1) //barrel
          {

            vLab[iLab + infoOffset] = float(true); iLab++; //isBarrel //7
            vLab[iLab + infoOffset] = PXBDetId(detIds[j]).layer(); iLab++;
            vLab[iLab + infoOffset] = PXBDetId(detIds[j]).ladder(); iLab++;
            vLab[iLab + infoOffset] = -1.0; iLab++;
            vLab[iLab + infoOffset] = -1.0; iLab++;
            vLab[iLab + infoOffset] = -1.0; iLab++;
            vLab[iLab + infoOffset] = PXBDetId(detIds[j]).module(); iLab++; //14

          }
          else
          {
            vLab[iLab + infoOffset] = float(false); iLab++; //isBarrel
            vLab[iLab + infoOffset] = -1.0; iLab++;
            vLab[iLab + infoOffset] = -1.0; iLab++;
            vLab[iLab + infoOffset] = PXFDetId(detIds[j]).side(); iLab++;
            vLab[iLab + infoOffset] = PXFDetId(detIds[j]).disk(); iLab++;
            vLab[iLab + infoOffset] = PXFDetId(detIds[j]).panel(); iLab++;
            vLab[iLab + infoOffset] = PXFDetId(detIds[j]).module(); iLab++;
          }

          //Module orientation
          float ax1  = siHits[j]->det()->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp(); //15
          float ax2  = siHits[j]->det()->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();

          vLab[iLab + infoOffset] = float(ax1<ax2); iLab++; //isFlipped
          vLab[iLab + infoOffset] = ax1; iLab++; //Module orientation y
          vLab[iLab + infoOffset] = ax2; iLab++; //Module orientation x

          auto thisCluster = siHits[j]->cluster();
          //TODO check CLusterRef & OmniClusterRef

          float xC = (float) thisCluster->x(), yC = (float) thisCluster->y();

  //inX
          vLab[iLab + infoOffset] = (float)xC; iLab++; //20
          vLab[iLab + infoOffset] = (float)yC; iLab++;
          vLab[iLab + infoOffset] = (float)thisCluster->size(); iLab++;
          vLab[iLab + infoOffset] = (float)thisCluster->sizeX(); iLab++;
          vLab[iLab + infoOffset] = (float)thisCluster->sizeY(); iLab++;
          vLab[iLab + infoOffset] = (float)thisCluster->pixel(0).adc; iLab++; //25
          vLab[iLab + infoOffset] = float(thisCluster->charge())/float(thisCluster->size()); iLab++; //avg pixel charge

          vLab[iLab + infoOffset] = (float)(thisCluster->sizeX() > padSize); iLab++;//27
          vLab[iLab + infoOffset] = (float)(thisCluster->sizeY() > padSize); iLab++;
          vLab[iLab + infoOffset] = (float)(thisCluster->sizeY()) / (float)(thisCluster->sizeX()); iLab++;

          vLab[iLab + infoOffset] = (float)siHits[j]->spansTwoROCs(); iLab++;
          vLab[iLab + infoOffset] = (float)siHits[j]->hasBadPixels(); iLab++;
          vLab[iLab + infoOffset] = (float)siHits[j]->isOnEdge(); iLab++; //31

          vLab[iLab + infoOffset] = (float)(thisCluster->charge()); iLab++;

          deltaA   -= ((float)thisCluster->size()); deltaA *= -1.0;
          deltaADC -= thisCluster->charge(); deltaADC *= -1.0; //At the end == Outer Hit ADC - Inner Hit ADC
          deltaS   -= ((float)(thisCluster->sizeY()) / (float)(thisCluster->sizeX())); deltaS *= -1.0;
          deltaR   -= copyDoublets.r(iD,layers[j]); deltaR *= -1.0;
          deltaPhi -= phi; deltaPhi *= -1.0;

          // TH2F hClust("hClust","hClust",
          // padSize,
          // thisCluster->x()-padSize/2,
          // thisCluster->x()+padSize/2,
          // padSize,
          // thisCluster->y()-padSize/2,
          // thisCluster->y()+padSize/2);
          //
          // //Initialization
          // for (int nx = 0; nx < padSize; ++nx)
          // for (int ny = 0; ny < padSize; ++ny)
          // hClust.SetBinContent(nx,ny,0.0);
          //
          // for (int k = 0; k < thisCluster->size(); ++k)
          // hClust.SetBinContent(hClust.FindBin((float)thisCluster->pixel(k).x, (float)thisCluster->pixel(k).y),(float)thisCluster->pixel(k).adc);
          //
          //
          // for (int ny = padSize; ny>0; --ny)
          // {
          //   for(int nx = 0; nx<padSize; nx++)
          //   {
          //     int n = (ny+2)*(padSize + 2) - 2 -2 - nx - padSize; //see TH2 reference for clarification
          //     hitPads[j].push_back(hClust.GetBinContent(n));
          //   }
          // }


          // //Pad Initialization
          // for (int iP = 0; iP < padSize*padSize*cnnLayers; ++iP)
          //   vPad[iP + doubOffset + j*padSize*padSize*cnnLayers] = 0.0;
          //
          // for (int k = 0; k < thisCluster->size(); ++k)
          // {
          //   int thisX = int(-(float)thisCluster->pixel(k).x + xC + padHalfSize);
          //   int thisY = int(-(float)thisCluster->pixel(k).y + yC + padHalfSize);
          //   vPad[padOffset + thisX + thisY * padSize + doubOffset] = (float)thisCluster->pixel(k).adc;
          // }


        }

        // for (int nx = 0; nx < padSize*padSize; ++nx)
        //     inHitPads[layerIds[0]][nx] = hitPads[0][nx];
        // for (int nx = 0; nx < padSize*padSize; ++nx)
        //     outHitPads[layerIds[1]][nx] = hitPads[1][nx];

        // std::cout << "Inner hit layer : " << innerLayer->seqNum() << " - " << layerIds[0]<< std::endl;
        //
        // for(int i = 0; i < cnnLayers; ++i)
        // {
        //   std::cout << i << std::endl;
        //   auto thisOne = inHitPads[i];
        //   for (int nx = 0; nx < padSize; ++nx)
        //     for (int ny = 0; ny < padSize; ++ny)
        //     {
        //       std::cout << thisOne[ny + nx*padSize] << " ";
        //     }
        //     std::cout << std::endl;
        //
        // }
        //
        // std::cout << "Outer hit layer : " << outerLayer->seqNum() << " - " << layerIds[1]<< std::endl;
        // for(int i = 0; i < cnnLayers; ++i)
        // {
        //   std::cout << i << std::endl;
        //   auto thisOne = outHitPads[i];
        //   for (int nx = 0; nx < padSize; ++nx)
        //     for (int ny = 0; ny < padSize; ++ny)
        //     {
        //       std::cout << thisOne[ny + nx*padSize ] << " ";
        //     }
        //     std::cout << std::endl;
        //
        // }
        //
        // std::cout << "TF Translation" << std::endl;
        // for(int i = 0; i < cnnLayers*2; ++i)
        // {
        //   std::cout << i << std::endl;
        //   int theOffset = i*padSize*padSize;
        //   for (int nx = 0; nx < padSize; ++nx)
        //     for (int ny = 0; ny < padSize; ++ny)
        //     {
        //       std::cout << vPad[(ny + nx*padSize) + theOffset + doubOffset] << " ";
        //     }
        //     std::cout << std::endl;
        //
        // }

        zZero = (siHits[0]->globalState()).position.z();
        zZero -= copyDoublets.r(iD,layers[0]) * (deltaZ/deltaR);

        vLab[iLab + infoOffset] = deltaA   ; iLab++;
        vLab[iLab + infoOffset] = deltaADC ; iLab++;
        vLab[iLab + infoOffset] = deltaS   ; iLab++;
        vLab[iLab + infoOffset] = deltaR   ; iLab++;
        vLab[iLab + infoOffset] = deltaPhi ; iLab++;
        vLab[iLab + infoOffset] = deltaZ   ; iLab++;
        vLab[iLab + infoOffset] = zZero    ; iLab++;

        // std::cout << "iLab = "<<iLab << std::endl;

      }
      // std::cout << "Making Inference" << std::endl;

      auto finishData = std::chrono::high_resolution_clock::now();

      auto startInf = std::chrono::high_resolution_clock::now();
      // tensorflow::run(session, { { "hit_shape_input", inputPads }, { "info_input", inputFeat } },
      //               { "output/Softmax" }, &outputs);
      tensorflow::run(session, { { "info_input", inputFeat } },
                    { "output/Softmax" }, &outputs);
      auto finishInf = std::chrono::high_resolution_clock::now();
      // std::cout << "Cleaning doublets" << std::endl;

      auto startPush = std::chrono::high_resolution_clock::now();
      copyDoublets.clear();
      float* score = outputs[0].flat<float>().data();
      for (int i = 0; i < numOfDoublets; i++)
        if(score[i*2 + 1]>t_)
          copyDoublets.add(inIndex[i],outIndex[i]);
      auto finishPush = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsedInf  = finishInf - startInf;
      std::chrono::duration<double> elapsedData = finishData - startData;
      std::chrono::duration<double> elapsedPush = finishPush - startPush;
      //
      // std::cout << "Staring size       : " << numOfDoublets << std::endl;
      // std::cout << "New size           : " << copyDoublets.size() << std::endl;
      // std::cout << "Elapsed time (data): " << elapsedData.count() << " s\n";
      // std::cout << "Elapsed time (inf) : " << elapsedInf.count() << " s\n";
      // std::cout << "Elapsed time (push): " << elapsedPush.count() << " s\n";

      return copyDoublets;

    }


    void produce(const bool clusterCheckOk, edm::Event& iEvent, const edm::EventSetup& iSetup) override {
      auto regionsLayers = regionsLayers_.beginEvent(iEvent);

      auto seedingHitSetsProducer = T_SeedingHitSets(&localRA_);
      auto intermediateHitDoubletsProducer = T_IntermediateHitDoublets(regionsLayers.seedingLayerSetsHitsPtr());

      if(!clusterCheckOk) {
        seedingHitSetsProducer.putEmpty(iEvent);
        intermediateHitDoubletsProducer.putEmpty(iEvent);
        return;
      }

      seedingHitSetsProducer.reserve(regionsLayers.regionsSize());
      intermediateHitDoubletsProducer.reserve(regionsLayers.regionsSize());

      for(const auto& regionLayers: regionsLayers) {
        const TrackingRegion& region = regionLayers.region();
        auto hitCachePtr_filler_shs = seedingHitSetsProducer.beginRegion(&region, nullptr);
        auto hitCachePtr_filler_ihd = intermediateHitDoubletsProducer.beginRegion(&region, std::get<0>(hitCachePtr_filler_shs));
        auto hitCachePtr = std::get<0>(hitCachePtr_filler_ihd);

        for(SeedingLayerSetsHits::SeedingLayerSet layerSet: regionLayers.layerPairs()) {
          auto doublets = generator_.doublets(region, iEvent, iSetup, layerSet, *hitCachePtr);
          LogTrace("HitPairEDProducer") << " created " << doublets.size() << " doublets for layers " << layerSet[0].index() << "," << layerSet[1].index();

          if(doublets.empty()) continue; // don't bother if no pairs from these layers

          if(doInference_ && layerSet[0].index() <10 && layerSet[0].index() > -1 && layerSet[1].index() < 10 && layerSet[1].index() > -1)
          {
            // std::cout << "HitPairEDProducer created " << doublets.size() << " doublets for layers " << layerSet[0].index() << "," << layerSet[1].index();
            auto cleanDoublets = cnnInference(doublets);
            seedingHitSetsProducer.fill(std::get<1>(hitCachePtr_filler_shs), cleanDoublets);
            intermediateHitDoubletsProducer.fill(std::get<1>(hitCachePtr_filler_ihd), layerSet, std::move(cleanDoublets));
          }else
          {
            seedingHitSetsProducer.fill(std::get<1>(hitCachePtr_filler_shs), doublets);
            intermediateHitDoubletsProducer.fill(std::get<1>(hitCachePtr_filler_ihd), layerSet, std::move(doublets));
          }

        }
      }

      seedingHitSetsProducer.put(iEvent);
      intermediateHitDoubletsProducer.put(iEvent);
    }

  private:
    T_RegionLayers regionsLayers_;
  };

  /////
  class DoNothing {
  public:
    DoNothing(const SeedingLayerSetsHits *) {}
    DoNothing(edm::RunningAverage *) {}

    static void produces(edm::ProducerBase&) {};

    void reserve(size_t) {}

    auto beginRegion(const TrackingRegion *, LayerHitMapCache *ptr) {
      return std::make_tuple(ptr, 0);
    }

    void fill(int, const HitDoublets&) {}
    void fill(int, const SeedingLayerSetsHits::SeedingLayerSet&, HitDoublets&&) {}

    void put(edm::Event&) {}
    void putEmpty(edm::Event&) {}
  };

  /////
  class ImplSeedingHitSets {
  public:
    ImplSeedingHitSets(edm::RunningAverage *localRA):
      seedingHitSets_(std::make_unique<RegionsSeedingHitSets>()),
      localRA_(localRA)
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<RegionsSeedingHitSets>();
    }

    void reserve(size_t regionsSize) {
      seedingHitSets_->reserve(regionsSize, localRA_->upper());
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *) {
      hitCacheTmp_.clear();
      return std::make_tuple(&hitCacheTmp_, seedingHitSets_->beginRegion(region));
    }

    void fill(RegionsSeedingHitSets::RegionFiller& filler, const HitDoublets& doublets) {
      for(size_t i=0, size=doublets.size(); i<size; ++i) {
        filler.emplace_back(doublets.hit(i, HitDoublets::inner),
                            doublets.hit(i, HitDoublets::outer));
      }
    }

    void put(edm::Event& iEvent) {
      seedingHitSets_->shrink_to_fit();
      localRA_->update(seedingHitSets_->size());
      putEmpty(iEvent);
    }

    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(seedingHitSets_));
    }

  private:
    std::unique_ptr<RegionsSeedingHitSets> seedingHitSets_;
    edm::RunningAverage *localRA_;
    LayerHitMapCache hitCacheTmp_; // used if !produceIntermediateHitDoublets
  };

  /////
  class ImplIntermediateHitDoublets {
  public:
    ImplIntermediateHitDoublets(const SeedingLayerSetsHits *layers):
      intermediateHitDoublets_(std::make_unique<IntermediateHitDoublets>(layers)),
      layers_(layers)
    {}

    static void produces(edm::ProducerBase& producer) {
      producer.produces<IntermediateHitDoublets>();
    }

    void reserve(size_t regionsSize) {
      intermediateHitDoublets_->reserve(regionsSize, layers_->size());
    }

    auto beginRegion(const TrackingRegion *region, LayerHitMapCache *) {
      auto filler = intermediateHitDoublets_->beginRegion(region);
      return std::make_tuple(&(filler.layerHitMapCache()), std::move(filler));
    }

    void fill(IntermediateHitDoublets::RegionFiller& filler, const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets) {
      filler.addDoublets(layerSet, std::move(doublets));
    }

    void put(edm::Event& iEvent) {
      intermediateHitDoublets_->shrink_to_fit();
      putEmpty(iEvent);
    }

    void putEmpty(edm::Event& iEvent) {
      iEvent.put(std::move(intermediateHitDoublets_));
    }

  private:
    std::unique_ptr<IntermediateHitDoublets> intermediateHitDoublets_;
    const SeedingLayerSetsHits *layers_;
  };

  /////
  // For the usual case that TrackingRegions and seeding layers are read separately
  class RegionsLayersSeparate {
  public:
    class RegionLayers {
    public:
      RegionLayers(const TrackingRegion *region, const std::vector<SeedingLayerSetsHits::SeedingLayerSet> *layerPairs):
        region_(region), layerPairs_(layerPairs) {}

      const TrackingRegion& region() const { return *region_; }
      const std::vector<SeedingLayerSetsHits::SeedingLayerSet>& layerPairs() const { return *layerPairs_; }

    private:
      const TrackingRegion *region_;
      const std::vector<SeedingLayerSetsHits::SeedingLayerSet> *layerPairs_;
    };

    class EventTmp {
    public:
      class const_iterator {
      public:
        using internal_iterator_type = edm::OwnVector<TrackingRegion>::const_iterator;
        using value_type = RegionLayers;
        using difference_type = internal_iterator_type::difference_type;

        const_iterator(internal_iterator_type iter, const std::vector<SeedingLayerSetsHits::SeedingLayerSet> *layerPairs):
          iter_(iter), layerPairs_(layerPairs) {}

        value_type operator*() const { return value_type(&(*iter_), layerPairs_); }

        const_iterator& operator++() { ++iter_; return *this; }
        const_iterator operator++(int) {
          const_iterator clone(*this);
          ++(*this);
          return clone;
        }

        bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
        bool operator!=(const const_iterator& other) const { return !operator==(other); }

      private:
        internal_iterator_type iter_;
        const std::vector<SeedingLayerSetsHits::SeedingLayerSet> *layerPairs_;
      };

      EventTmp(const SeedingLayerSetsHits *seedingLayerSetsHits,
               const edm::OwnVector<TrackingRegion> *regions,
               const std::vector<unsigned>& layerPairBegins):
        seedingLayerSetsHits_(seedingLayerSetsHits), regions_(regions) {

        // construct the pairs from the sets
        if(seedingLayerSetsHits_->numberOfLayersInSet() > 2) {
          for(const auto& layerSet: *seedingLayerSetsHits_) {
            for(const auto pairBeginIndex: layerPairBegins) {
              if(pairBeginIndex+1 >= seedingLayerSetsHits->numberOfLayersInSet()) {
                throw cms::Exception("LogicError") << "Layer pair index " << pairBeginIndex << " is out of bounds, input SeedingLayerSetsHits has only " << seedingLayerSetsHits->numberOfLayersInSet() << " layers per set, and the index+1 must be < than the number of layers in set";
              }

              // Take only the requested pair of the set
              SeedingLayerSetsHits::SeedingLayerSet pairCandidate = layerSet.slice(pairBeginIndex, pairBeginIndex+1);

              // it would be trivial to use 128-bit bitfield for O(1) check
              // if a layer pair has been inserted, but let's test first how
              // a "straightforward" solution works
              auto found = std::find_if(layerPairs.begin(), layerPairs.end(), [&](const SeedingLayerSetsHits::SeedingLayerSet& pair) {
                  return pair[0].index() == pairCandidate[0].index() && pair[1].index() == pairCandidate[1].index();
                });
              if(found != layerPairs.end())
                continue;

              layerPairs.push_back(pairCandidate);
            }
          }
        }
        else {
          if(layerPairBegins.size() != 1) {
            throw cms::Exception("LogicError") << "With pairs of input layers, it doesn't make sense to specify more than one input layer pair, got " << layerPairBegins.size();
          }
          if(layerPairBegins[0] != 0) {
            throw cms::Exception("LogicError") << "With pairs of input layers, it doesn't make sense to specify other input layer pair than 0; got " << layerPairBegins[0];
          }

          layerPairs.reserve(seedingLayerSetsHits->size());
          for(const auto& set: *seedingLayerSetsHits)
            layerPairs.push_back(set);
        }
      }

      const SeedingLayerSetsHits *seedingLayerSetsHitsPtr() const { return seedingLayerSetsHits_; }

      size_t regionsSize() const { return regions_->size(); }

      const_iterator begin() const { return const_iterator(regions_->begin(), &layerPairs); }
      const_iterator cbegin() const { return begin(); }
      const_iterator end() const { return const_iterator(regions_->end(), &layerPairs); }
      const_iterator cend() const { return end(); }

    private:
      const SeedingLayerSetsHits *seedingLayerSetsHits_;
      const edm::OwnVector<TrackingRegion> *regions_;
      std::vector<SeedingLayerSetsHits::SeedingLayerSet> layerPairs;
    };

    RegionsLayersSeparate(const std::vector<unsigned> *layerPairBegins,
                          const edm::InputTag& seedingLayerTag,
                          const edm::InputTag& regionTag,
                          edm::ConsumesCollector&& iC):
      layerPairBegins_(layerPairBegins),
      seedingLayerToken_(iC.consumes<SeedingLayerSetsHits>(seedingLayerTag)),
      regionToken_(iC.consumes<edm::OwnVector<TrackingRegion> >(regionTag))
    {}

    EventTmp beginEvent(const edm::Event& iEvent) const {
      edm::Handle<SeedingLayerSetsHits> hlayers;
      iEvent.getByToken(seedingLayerToken_, hlayers);
      const auto *layers = hlayers.product();
      if(layers->numberOfLayersInSet() < 2)
        throw cms::Exception("LogicError") << "HitPairEDProducer expects SeedingLayerSetsHits::numberOfLayersInSet() to be >= 2, got " << layers->numberOfLayersInSet() << ". This is likely caused by a configuration error of this module, or SeedingLayersEDProducer.";
      edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
      iEvent.getByToken(regionToken_, hregions);

      return EventTmp(layers, hregions.product(), *layerPairBegins_);
    }

  private:
    const std::vector<unsigned> *layerPairBegins_;
    edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
    edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;
  };

  /////
  // For the case of automated pixel inactive region recovery where
  // TrackingRegions and seeding layers become together
  class RegionsLayersTogether {
  public:
    class EventTmp {
    public:
      using const_iterator = TrackingRegionsSeedingLayerSets::const_iterator;

      explicit EventTmp(const TrackingRegionsSeedingLayerSets *regionsLayers):
        regionsLayers_(regionsLayers) {
        if(regionsLayers->seedingLayerSetsHits().numberOfLayersInSet() != 2) {
          throw cms::Exception("LogicError") << "With trackingRegionsSeedingLayers input, the seeding layer sets may only contain layer pairs, now got sets with " << regionsLayers->seedingLayerSetsHits().numberOfLayersInSet() << " layers";
        }
      }

      const SeedingLayerSetsHits *seedingLayerSetsHitsPtr() const { return &(regionsLayers_->seedingLayerSetsHits()); }

      size_t regionsSize() const { return regionsLayers_->regionsSize(); }

      const_iterator begin() const { return regionsLayers_->begin(); }
      const_iterator cbegin() const { return begin(); }
      const_iterator end() const { return regionsLayers_->end(); }
      const_iterator cend() const { return end(); }

    private:
      const TrackingRegionsSeedingLayerSets *regionsLayers_;
    };

    RegionsLayersTogether(const std::vector<unsigned> *layerPairBegins,
                          const edm::InputTag& regionLayerTag,
                          edm::ConsumesCollector&& iC):
      regionLayerToken_(iC.consumes<TrackingRegionsSeedingLayerSets>(regionLayerTag))
    {
      if(layerPairBegins->size() != 1) {
        throw cms::Exception("LogicError") << "With trackingRegionsSeedingLayers mode, it doesn't make sense to specify more than one input layer pair, got " << layerPairBegins->size();
      }
      if((*layerPairBegins)[0] != 0) {
        throw cms::Exception("LogicError") << "With trackingRegionsSeedingLayers mode, it doesn't make sense to specify other input layer pair than 0; got " << (*layerPairBegins)[0];
      }
    }

    EventTmp beginEvent(const edm::Event& iEvent) const {
      edm::Handle<TrackingRegionsSeedingLayerSets> hregions;
      iEvent.getByToken(regionLayerToken_, hregions);
      return EventTmp(hregions.product());
    }

  private:
    edm::EDGetTokenT<TrackingRegionsSeedingLayerSets> regionLayerToken_;
  };
}



HitPairEDProducer::HitPairEDProducer(const edm::ParameterSet& iConfig) {
  auto layersTag = iConfig.getParameter<edm::InputTag>("seedingLayers");
  auto regionTag = iConfig.getParameter<edm::InputTag>("trackingRegions");
  auto regionLayerTag = iConfig.getParameter<edm::InputTag>("trackingRegionsSeedingLayers");
  const bool useRegionLayers = regionLayerTag.label() != "";
  if(useRegionLayers) {
    if(regionTag.label() != "") {
      throw cms::Exception("Configuration") << "HitPairEDProducer requires either trackingRegions or trackingRegionsSeedingLayers to be set, now both are set to non-empty value. Set the unneeded parameter to empty value.";
    }
    if(layersTag.label() != "") {
      throw cms::Exception("Configuration") << "With non-empty trackingRegionsSeedingLayers, please set also seedingLayers to empty value to reduce confusion, because the parameter is not used";
    }
  }
  if(regionTag.label() == "" && regionLayerTag.label() == "") {
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either trackingRegions or trackingRegionsSeedingLayers to be set, now both are set to empty value. Set the needed parameter to a non-empty value.";
  }

  const bool produceSeedingHitSets = iConfig.getParameter<bool>("produceSeedingHitSets");
  const bool produceIntermediateHitDoublets = iConfig.getParameter<bool>("produceIntermediateHitDoublets");

  if(produceSeedingHitSets && produceIntermediateHitDoublets) {
    if(useRegionLayers) throw cms::Exception("Configuration") << "Mode 'trackingRegionsSeedingLayers' makes sense only with 'produceSeedingHitsSets', now also 'produceIntermediateHitDoublets is active";
    impl_ = std::make_unique<::Impl<::ImplSeedingHitSets, ::ImplIntermediateHitDoublets, ::RegionsLayersSeparate>>(iConfig, layersTag, regionTag, consumesCollector());
  }
  else if(produceSeedingHitSets) {
    if(useRegionLayers) {
      impl_ = std::make_unique<::Impl<::ImplSeedingHitSets, ::DoNothing, ::RegionsLayersTogether>>(iConfig, regionLayerTag, consumesCollector());
    }
    else {
      impl_ = std::make_unique<::Impl<::ImplSeedingHitSets, ::DoNothing, ::RegionsLayersSeparate>>(iConfig, layersTag, regionTag, consumesCollector());
    }
  }
  else if(produceIntermediateHitDoublets) {
    if(useRegionLayers) throw cms::Exception("Configuration") << "Mode 'trackingRegionsSeedingLayers' makes sense only with 'produceSeedingHitsSets', now 'produceIntermediateHitDoublets is active instead";
    impl_ = std::make_unique<::Impl<::DoNothing, ::ImplIntermediateHitDoublets, ::RegionsLayersSeparate>>(iConfig, layersTag, regionTag, consumesCollector());
  }
  else
    throw cms::Exception("Configuration") << "HitPairEDProducer requires either produceIntermediateHitDoublets or produceSeedingHitSets to be True. If neither are needed, just remove this module from your sequence/path as it doesn't do anything useful";

  auto clusterCheckTag = iConfig.getParameter<edm::InputTag>("clusterCheck");
  if(clusterCheckTag.label() != "")
    clusterCheckToken_ = consumes<bool>(clusterCheckTag);

  impl_->produces(*this);
}

void HitPairEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("seedingLayers", edm::InputTag("seedingLayersEDProducer"))->setComment("Set this empty if 'trackingRegionsSeedingLayers' is non-empty");
  desc.add<edm::InputTag>("trackingRegions", edm::InputTag("globalTrackingRegionFromBeamSpot"))->setComment("Input tracking regions when using all layer sets in 'seedingLayers' (conflicts with 'trackingRegionsSeedingLayers', set this empty to use the other)");
  desc.add<edm::InputTag>("trackingRegionsSeedingLayers", edm::InputTag(""))->setComment("Input tracking regions and corresponding layer sets in case of dynamically limiting the seeding layers (conflicts with 'trackingRegions', set this empty to use the other; if using this set also 'seedingLayers' to empty)");
  desc.add<edm::InputTag>("clusterCheck", edm::InputTag("trackerClusterCheck"));
  desc.add<bool>("produceSeedingHitSets", false);
  desc.add<bool>("produceIntermediateHitDoublets", false);
  desc.add<unsigned int>("maxElement", 1000000);
  desc.add<std::vector<unsigned> >("layerPairs", std::vector<unsigned>{0})->setComment("Indices to the pairs of consecutive layers, i.e. 0 means (0,1), 1 (1,2) etc.");

  descriptions.add("hitPairEDProducerDefault", desc);
}

void HitPairEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool clusterCheckOk = true;
  if(!clusterCheckToken_.isUninitialized()) {
    edm::Handle<bool> hclusterCheck;
    iEvent.getByToken(clusterCheckToken_, hclusterCheck);
    clusterCheckOk = *hclusterCheck;
  }

  impl_->produce(clusterCheckOk, iEvent, iSetup);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HitPairEDProducer);
